#include "cuda.cuh"
#include "array.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>

float pi() { return std::atan(1)*4; }

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

__managed__ float deltaT = 0.001f;

__managed__ float tau = 0.0007;
__managed__ float cellwidth = .01f;

__managed__ bool changes = false;

bool fanStatus = false;
bool desiredFanStatus = false;

bool pause = false;

__managed__ float currentTime;

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

__device__ __host__ inline float feq(size_t i, float p, const vec3f& v) {
    float wi = wis[i];
    float c = cellwidth;
    float dot = offsets[i] * c * v;
    return wi * p * (1 + (1 / (c * c)) * (3 * dot + (9 / (2 * c * c)) * dot * dot - (3.f / 2) * (v * v)));
}

__device__ inline void collisionStep(cell_t &cell) {
    float p = 0;
    float c = cellwidth;
    floatparts* parts = (floatparts*) &cell[0];
    if (parts->exponent == 255) {
        if ((parts->mantissa & 1) != 0) {
            for (size_t i = 1; i < Q; i++) {
                cell[i] = cell[opposite[i]];
            }
        }
        return;
    }
    vec3f vp {0, 0, 0};
    for (size_t i = 0; i < Q; i++) {
        p += cell[i];
        vp += offsets[i] * c * cell[i];
    }
    vec3f v = p == 0 ? vp : vp * (1 / p);

    for (size_t i = 0; i < Q; i++) {
        cell[i] = cell[i] + deltaT / tau * (feq(i, p, v) - cell[i]);
    }
}

__global__ void updateCollision(cell_t *src, vec3<size_t> size, size_t zoffset, size_t zsize) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + zoffset;
    if (x >= size.x || y >= size.y || z >= zsize + zoffset) {
        return;
    }
    size_t i = pack(size.x, size.y, size.z, x, y, z);
    collisionStep(src[i]);
}

__global__ void updateStreaming(cell_t *dst, cell_t *src, vec3<size_t> size, size_t zoffset, size_t zsize) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + zoffset;
    if (x >= size.x || y >= size.y || z >= zsize + zoffset) {
        return;
    }
    size_t index = pack(size.x, size.y, size.z, x, y, z);

    floatparts* parts = (floatparts*) &src[index][0];

    if (parts->exponent == 255) {
        return;
    }

    for (int i = 1; i < Q; i++) {
        int sx = x + (int) offsets[i].x;
        int sy = y + (int) offsets[i].y;
        int sz = z + (int) offsets[i].z;
        if (sx < 0 || sy < 0 || sz < 0 || sx >= size.x || sy >= size.y || sz >= size.z) {
            continue;
        }
        dst[index][i] = src[pack(size.x, size.y, size.z, sx, sy, sz)][i];
    }
}

__device__ unsigned char floatToChar(float f) {
    return (unsigned char) min(max((f * 100.f + 1.f) * 127.f, 0.f), 255.f);
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

void setTime(float _time) {
    currentTime = _time;
}

void initSimulation(size_t xdim, size_t ydim, size_t zdim, size_t gpus) {
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

    for (int x = 0; x < size.x; x++) {
        for (int y = 0; y < size.y; y++) {
            for (int z = 0; z < size.z; z++) {
                for (int i = 0; i < Q; i++) {
                    float f = feq(i, 0.1f, {.001f, 0, 0});
                    u1[pack(size.x, size.y, size.z, x, y, z)][i] = f;
                    u2[pack(size.x, size.y, size.z, x, y, z)][i] = f;
                }

                if (x <= 1 || y <= 1 || z <= 1 || x >= size.x - 2 || y >= size.y - 2 || z >= size.y - 2 ||  //xdim == 20 && (ydim >= 40 && ydim <= 48 || ydim >= 52 && ydim <= 60) || xdim == 23 && ydim == 51) {
                    std::pow(x - 50, 2) + std::pow(y - 50, 2) + std::pow(z - 8, 2) <= 225) {
                    auto* parts = (floatparts*) &u1[pack(size.x, size.y, size.z, x, y, z)][0];
                    parts->sign = 0;
                    parts->exponent = 255;
                    if (x <= 1 || x >= size.x - 2 || y <= 1 || y >= size.y - 2) {
                        parts->mantissa = 1 << 22 | 0b10;
                    } else {
                        parts->mantissa = 1 << 22 | 0b01;
                    }
                    u2[pack(size.x, size.y, size.z, x, y, z)][0] = u1[pack(size.x, size.y, size.z, x, y, z)][0];
                }
            }
        }
    }

    for (auto gpu : gpuStructs) {
        gpuErrchk(cudaMemcpyAsync(&gpu.data1[gpu.mainOffset], &u1[gpu.mainGlobalIndex], gpu.mainLayers * bytesPerLayer, cudaMemcpyDefault, streams[gpu.device]));
        gpuErrchk(cudaMemcpyAsync(&gpu.data2[gpu.mainOffset], &u2[gpu.mainGlobalIndex], gpu.mainLayers * bytesPerLayer, cudaMemcpyDefault, streams[gpu.device]));
    }
    syncStreams();
}

void togglePause() {
    pause = !pause;
}

void setFan(bool fan) {
    desiredFanStatus = fan;
}

bool getFan() {
    return desiredFanStatus;
}

void simulateStep() {
    if (pause) {
        return;
    }

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

    for (auto gpu : gpuStructs) {
        cudaSetDevice(gpu.device);
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks(
                (size.x + threadsPerBlock.x) / threadsPerBlock.x,
                (size.y + threadsPerBlock.y) / threadsPerBlock.y,
                (gpu.mainLayers + threadsPerBlock.z) / threadsPerBlock.z
        );
        updateCollision<<<numBlocks, threadsPerBlock, 0, streams[gpu.device]>>>(gpu.data1, size, gpu.mainOffset / elementsPerLayer, gpu.mainLayers);
        updateStreaming<<<numBlocks, threadsPerBlock, 0, streams[gpu.device]>>>(gpu.data2, gpu.data1, size, gpu.mainOffset / elementsPerLayer, gpu.mainLayers);
        // data1 is always pointing to up-to-date buffer.
        std::swap(gpu.data1, gpu.data2);
    }
    std::swap(u1, u2);
    syncStreams();
}

void printLayer(size_t z) {
    gpuErrchk(cudaMemcpy(u1, gpuStructs[0].data1, sizeof(cell_t) * cells, cudaMemcpyDeviceToHost));

    for (size_t y = 0; y < 5u; y++) {
        for (size_t x = 0; x < 5u; x++) {
            cell_t v = u1[pack(size.x, size.y, size.z, x, y, z)];
            printf("(%f,%f,%f), ", v[0], v[1], v[2]);
        }
        printf("\n");
    }
    printf("\n");

}

void exportFrame(const std::string& filename) {
    std::ofstream out;
    out.open(filename, std::ios::out | std::ios::binary);

    for (auto gpu : gpuStructs) {
        gpuErrchk(cudaMemcpy(&u1[gpu.mainGlobalIndex], &gpu.data1[gpu.mainOffset], gpu.mainLayers * bytesPerLayer, cudaMemcpyDeviceToHost));
        out.write(reinterpret_cast<const char *>(&u1[gpu.mainGlobalIndex]), gpu.mainLayers * bytesPerLayer);
    }

    out.close();
}

void importFrame() {
/*
    turnOffFan();

    std::ifstream in;
    in.open("scenario.dat", std::ios::in | std::ios::binary);

    for (int x = 1; x < SIZE.x - 1; x++) {
        for (int y = 1; y < SIZE.y - 1; y++) {
            for (int z = 1; z < SIZE.z - 1; z++) {
                int i = pack(SIZE.x, SIZE.y, SIZE.z, x, y, z);
                in.read(reinterpret_cast<char *>(&u1[i].x), sizeof(float));
                in.read(reinterpret_cast<char *>(&u1[i].y), sizeof(float));
                in.read(reinterpret_cast<char *>(&u1[i].z), sizeof(float));
                in.read(reinterpret_cast<char *>(&p1[i]), sizeof(float));
            }
        }
    }
    gpuErrchk(cudaMemcpy(cudau1, u1, sizeof(glm::vec3) * CELLS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudap1, p1, sizeof(float) * CELLS, cudaMemcpyHostToDevice));
    in.close();
    */
}

