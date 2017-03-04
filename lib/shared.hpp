#ifndef PAR_SHARED_HPP
#define PAR_SHARED_HPP

#define ACCELERATE_MODE_NONE 0
#define ACCELERATE_MODE_CUDA 1
#define ACCELERATE_MODE_OPENCL 2

#define ACCELERATE_MODE ACCELERATE_MODE_OPENCL

#if ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
#include <clBLAS.h>
#define MAX_NUM_DEVICES 16
#define MAX_DEVICE_NAME 1024
#define CURRENT_DEVICE 1

extern cl_context cl_ctx;
extern cl_command_queue cl_queue;

cl_context cl_ctx;
cl_command_queue cl_queue;
#endif

void event_start() {
#if ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
  clblasSetup();

  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_device_id devices[MAX_NUM_DEVICES];
  cl_uint numDevices = 0;
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};

  err = clGetPlatformIDs(1, &platform, NULL);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  assert(numDevices == 2);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
  device = devices[CURRENT_DEVICE];
  char *value;
  size_t valueSize;
  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
  value = (char *)malloc(valueSize);
  clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
  printf("Device: %s\n", value);
  free(value);
  props[1] = (cl_context_properties)platform;
  cl_ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  cl_queue = clCreateCommandQueue(cl_ctx, device, 0, &err);
#endif
}

void event_stop() {
#if ACCELERATE_MODE == ACCELERATE_MODE_OPENCL
  clblasTeardown();
  clReleaseCommandQueue(cl_queue);
  clReleaseContext(cl_ctx);
#endif
}

#endif
