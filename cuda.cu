#include "cuda.cuh"
#include "array.h"
#include <cmath>
#include <fstream>
#include <vector>

Timer timer = Timer();
double time_split;

const int FLAG_OBSTACLE = 1 << 0;
const int FLAG_KEEP_VELOCITY = 1 << 1;

typedef struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
} floatparts;

const size_t Q = 19;
typedef array<float, Q> cell_t;
typedef vec3<float> vec3f;

struct gpu_t {
    int device;
    size_t mainGlobalIndex;
    size_t mainLayers;
    cell_t *data1;
    cell_t *data2;
    size_t mainOffset;
    size_t bottomPaddingOffset;
};

vec3<size_t> size;

size_t cells;
size_t bytesPerLayer;
size_t elementsPerLayer;

std::vector<gpu_t> gpuStructs;

cudaStream_t *streams;

__managed__ float deltaT = 1.f;

__managed__ float tau = 0.65;
__managed__ float cellwidth = 1.f;

bool pause = false;

__constant__ const array<vec3f, Q> offsets {
        0, 0, 0,   // 0
        -1, 0, 0,  // 1
        1, 0, 0,   // 2
        0, -1, 0,  // 3
        0, 1, 0,   // 4
        0, 0, -1,  // 5
        0, 0, 1,   // 6
        -1, -1, 0, // 7
        -1, 1, 0,  // 8
        1, -1, 0,  // 9
        1, 1, 0,   // 10
        -1, 0, -1, // 11
        -1, 0, 1,  // 12
        1, 0, -1,  // 13
        1, 0, 1,   // 14
        0, -1, -1, // 15
        0, -1, 1,  // 16
        0, 1, -1,  // 17
        0, 1, 1,   // 18
};

__constant__ const array<unsigned char, Q> opposite = {
        0,
        2, 1, 4, 3, 6, 5,
        10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
};

__constant__ const array<float, Q> wis {
    1.f / 3,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
};

cell_t *u1;
cell_t *u2;

__device__ __host__ inline size_t pack(size_t w, size_t h, size_t d, size_t x, size_t y, size_t z) {
    return (z * h + y) * w + x;
}

__device__ __host__ inline float feq(const size_t i, const float p, const vec3f& v) {
    float wi = wis[i];
    float c = cellwidth;
    float dot = offsets[i] * c * v;
    return wi * p * (1 + (1 / (c * c)) * (3 * dot + (9 / (2 * c * c)) * dot * dot - (3.f / 2) * (v * v)));
}

__device__ inline void collisionStep(cell_t &cell) {
    floatparts* parts = (floatparts*) &cell[0];
    if (parts->exponent == 255) {
        if (parts->mantissa & FLAG_OBSTACLE) {
            cell_t cell2 = cell;
            for (size_t i = 1; i < Q; i++) {
                cell[i] = cell2[opposite[i]];
            }
        }
        return;
    }
    float p = 0;
    vec3f vp {0, 0, 0};
    for (size_t i = 0; i < Q; i++) {
        p += cell[i];
        vp += offsets[i] * cellwidth * cell[i];
    }
    vec3f v = p == 0 ? vp : vp * (1 / p);

    for (size_t i = 0; i < Q; i++) {
        cell[i] = cell[i] + deltaT / tau * (feq(i, p, v) - cell[i]);
    }
}

__global__ void update(cell_t *dst, cell_t *src, const size_t worksize, const vec3<size_t> globalsize, const size_t zoffset) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= worksize) {
        return;
    }
    size_t x = i % globalsize.x;
    size_t y = (i / globalsize.x) % globalsize.y;
    size_t z = (i / (globalsize.x * globalsize.y)) + zoffset;
    size_t index = i + zoffset * globalsize.x * globalsize.y;

    cell_t dest = src[index];
    floatparts* parts = (floatparts*) &dest;

    if (parts->exponent == 255) {
        if (parts->mantissa & FLAG_KEEP_VELOCITY) {
            dst[index] = src[index];
            return;
        }
    }

    for (int i = 1; i < Q; i++) {
        int sx = x + (int) offsets[i].x;
        int sy = y + (int) offsets[i].y;
        int sz = z + (int) offsets[i].z;
        if (sx < 0 || sy < 0 || sz < 0 || sx >= globalsize.x || sy >= globalsize.y || sz >= globalsize.z) {
            continue;
        }
        dest[i] = src[pack(globalsize.x, globalsize.y, globalsize.z, sx, sy, sz)][i];
    }

    collisionStep(dest);
    dst[index] = dest;
}

__device__ unsigned char floatToChar(float f) {
    return (unsigned char) min(max((f * 10.f + 1.f) * 127.f, 0.f), 255.f);
}

void syncStreams() {
    for (auto gpu : gpuStructs) {
        gpuErrchk(cudaStreamSynchronize(streams[gpu.device]));
    }
}

__global__ void renderToBuffer(uchar4 *destImg, cell_t *srcU, vec3<size_t> size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x; // Not calculating border cells.
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = 8;

    size_t iP = pack(size.x, size.y, size.z, x, y, z);
    size_t iI = (size.y - y - 1) * size.x + x; // Invert opengl image.
    vec3f p{};
    cell_t cell = srcU[iP];
    for (int i = 0; i < Q; i++) {
        p += offsets[i] * cell[i];
    }
    destImg[iI] = {
            floatToChar(p.x), floatToChar(p.y), floatToChar(p.z), 255
    };

}

void render(uchar4 *img, const int width, const int height) {
    simulateStep();
    dim3 threadsPerBlock(1, 1);
    dim3 numBlocks(size.x, size.y);
    renderToBuffer<<<numBlocks, threadsPerBlock>>>(img, gpuStructs[0].data1, size);
    cudaDeviceSynchronize();
}

__device__ __host__ inline void generate(cell_t &cell, int x, int y, int z, const vec3<size_t> globalSize) {
    for (int i = 0; i < Q; i++) {
        float f = feq(i, 1.f, {.1f, 0, 0});
        cell[i] = f;
    }

    if (x <= 1 || y <= 1 || z <= 1 || x >= globalSize.x - 2 || y >= globalSize.y - 2 || z >= globalSize.z - 2 ||
        std::pow(x - 50, 2) + std::pow(y - 50, 2) + std::pow(z - 8, 2) <= 225) {
        auto *parts = (floatparts *) &cell[0];
        parts->sign = 0;
        parts->exponent = 255;
        if (x <= 1 || x >= globalSize.x - 2 || y <= 1 || y >= globalSize.y - 2 || z <= 1 || z >= globalSize.z - 2) {
            parts->mantissa = 1 << 22 | FLAG_KEEP_VELOCITY;
        } else {
            parts->mantissa = 1 << 22 | FLAG_OBSTACLE;
        }
    }
}


__global__ void init(cell_t *dst, size_t worksize, const vec3<size_t> globalsize, size_t zpaddingtop, size_t zoffset) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= worksize) {
        return;
    }

    size_t x = i % globalsize.x;
    size_t y = (i / globalsize.x) % globalsize.y;
    size_t z = (i / (globalsize.x * globalsize.y)) + zoffset;
    size_t index = i + zpaddingtop * globalsize.x * globalsize.y;

    generate(dst[index], x, y, z, globalsize);
}

void initSimulation(size_t xdim, size_t ydim, size_t zdim, size_t gpus, const std::string &importFile) {
    size = {xdim, ydim, zdim};
    cells = xdim * ydim * zdim;

    size_t layersPerGpu = zdim / gpus;
    size_t remainder = zdim - layersPerGpu * gpus;
    elementsPerLayer = xdim * ydim;
    bytesPerLayer = elementsPerLayer * sizeof(cell_t);

    u1 = new cell_t[cells];
    u2 = new cell_t[cells];

    streams = new cudaStream_t[gpus];

    size_t currentLayer = 0;
    gpuStructs = std::vector<gpu_t>();
    gpuStructs.reserve(gpus);

    for (int i = 0; i < gpus; i++) {
        gpuErrchk(cudaSetDevice(i));
        gpuErrchk(cudaStreamCreate(&streams[i]));
        gpu_t gpu{};
        gpu.device = i;
        int toppaddinglayers = i > 0 ? 1 : 0;
        int bottompaddinglayers = i < gpus - 1 ? 1 : 0;
        gpu.mainGlobalIndex = currentLayer * elementsPerLayer;
        gpu.mainLayers = layersPerGpu + (i < remainder ? 1 : 0);
        gpu.mainOffset = toppaddinglayers * elementsPerLayer;
        gpu.bottomPaddingOffset = gpu.mainOffset + gpu.mainLayers * elementsPerLayer;

        currentLayer += gpu.mainLayers;

        gpuErrchk(cudaMalloc(&gpu.data1, (gpu.mainLayers + toppaddinglayers + bottompaddinglayers) * bytesPerLayer));
        gpuErrchk(cudaMalloc(&gpu.data2, (gpu.mainLayers + toppaddinglayers + bottompaddinglayers) * bytesPerLayer));
        gpuStructs.push_back(gpu);
    }

    if (!importFile.empty()) {
        importFrame(importFile);
        for (auto gpu : gpuStructs) {
            gpuErrchk(cudaMemcpyAsync(&gpu.data1[gpu.mainOffset], &u1[gpu.mainGlobalIndex], gpu.mainLayers * bytesPerLayer, cudaMemcpyDefault, streams[gpu.device]));
        }
    } else {
        for (auto &gpu : gpuStructs) {
            cudaSetDevice(gpu.device);
            dim3 threadsPerBlock(512);
            size_t worksize = size.x * size.y * gpu.mainLayers;
            dim3 numBlocks(
                    (worksize + threadsPerBlock.x - 1) / threadsPerBlock.x
            );
            init<<<numBlocks, threadsPerBlock, 0, streams[gpu.device]>>>(
                    gpu.data1, worksize, size, gpu.mainOffset / elementsPerLayer, gpu.mainGlobalIndex / elementsPerLayer
            );
        }
    }
    syncStreams();
}

void togglePause() {
    pause = !pause;
}

void updateHost() {
    for (auto gpu : gpuStructs) {
        gpuErrchk(cudaMemcpy(&u1[gpu.mainGlobalIndex], &gpu.data1[gpu.mainOffset], gpu.mainLayers * bytesPerLayer,
                             cudaMemcpyDeviceToHost));
    }
}

void updateDevice() {
    for (auto gpu: gpuStructs) {
        gpuErrchk(cudaMemcpy(&gpu.data1[gpu.mainOffset], &u1[gpu.mainGlobalIndex], gpu.mainLayers * bytesPerLayer,
                cudaMemcpyHostToDevice));
    }
}
void syncMPI(){

}
void simulateStep() {
    if (pause) {
        return;
    }
    syncMPI();
    for (int i = 1; i < gpuStructs.size(); i++) {
        // Copy bottom padding for i - 1
        gpuErrchk(cudaMemcpyAsync(
                &gpuStructs[i - 1].data1[gpuStructs[i - 1].bottomPaddingOffset],
                &gpuStructs[i].data1[gpuStructs[i].mainOffset],
                bytesPerLayer, cudaMemcpyDefault, streams[i - 1]
        ));
        // Copy top padding for i
        gpuErrchk(cudaMemcpyAsync(
                gpuStructs[i].data1,
                &gpuStructs[i - 1].data1[gpuStructs[i - 1].bottomPaddingOffset - elementsPerLayer],
                bytesPerLayer, cudaMemcpyDefault, streams[i]
        ));
    }

    syncStreams();
    time_split = timer.get();
    for (auto &gpu : gpuStructs) {
        cudaSetDevice(gpu.device);
        dim3 threadsPerBlock(1024);
        size_t worksize = size.x * size.y * gpu.mainLayers;
        dim3 numBlocks(
                (worksize + threadsPerBlock.x - 1) / threadsPerBlock.x
        );
        update<<<numBlocks, threadsPerBlock, 0, streams[gpu.device]>>>(
                gpu.data2, gpu.data1, worksize, size, gpu.mainOffset / elementsPerLayer
        );
        std::swap(gpu.data1, gpu.data2); // data1 is always pointing to up-to-date buffer.
    }
    std::swap(u1, u2);
    syncStreams();
}

void exportFrame(const std::string& filename) {

    updateHost();

    std::ofstream out;
    out.open(filename, std::ios::out | std::ios::binary);

    for (auto gpu : gpuStructs) {
        gpuErrchk(cudaMemcpy(&u1[gpu.mainGlobalIndex], &gpu.data1[gpu.mainOffset], gpu.mainLayers * bytesPerLayer, cudaMemcpyDeviceToHost));
    }

    out.write(reinterpret_cast<const char *>(u1), cells * sizeof(cell_t));

    out.close();
}

void importFrame(const std::string& importFile) {
    std::ifstream infile(importFile, std::ios_base::binary);
    infile.read(reinterpret_cast<char *>(u1), cells * sizeof(cell_t));
    updateDevice();
}

