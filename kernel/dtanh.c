__kernel void dtanh(__global const float * x, __global const float * T, __global float * y)
{
   int row = get_global_id(0);
   float th = T[row];
   y[row] = (1.0-th*th) * x[row];
}