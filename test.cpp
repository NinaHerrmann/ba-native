#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Timer.h"
#include "cuda.cuh"

static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

void exitWithUsage() {
    std::cerr << "Usage: ./test [-d <xdim> <ydim> <zdim>] [-g <nGPUs>] [-n <iterations>] [-i <importFile] [-e <exportFile>]" << std::endl;
    exit(-1);
}

int getIntArg(char* s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        exitWithUsage();
    }
    return i;
}

int main(int argc, char** argv) {
    vec3<int> size {100, 100, 100};
    int gpus = 1;
    int iterations = 1;
    std::string importFile, exportFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch(argv[i++][1]) {
            case 'd':
                if (argc < i + 3) {
                    exitWithUsage();
                }
                size.x = getIntArg(argv[i++]);
                size.y = getIntArg(argv[i++]);
                size.z = getIntArg(argv[i]);
                break;
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'i':
                importFile = std::string(argv[i]);
                break;
            case 'e':
                exportFile = std::string(argv[i]);
                break;
            default:
                exitWithUsage();
        }
    }

    initSimulation(size.x, size.y, size.z, gpus, importFile);

    Timer t = Timer();

    t.start();

    for (int i = 0; i < iterations; i++) {
        simulateStep();
        std::cout << t.round() << std::endl;
    }

    if (!exportFile.empty()) {
        exportFrame(exportFile);
    }

}