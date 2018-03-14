__kernel void concatenate(__global const float * x1,__global const float * x2, __global float * y, int offset)
{
   int col = get_global_id(0);
   y[col] = col<offset ? x1[col] : x2[col];
}