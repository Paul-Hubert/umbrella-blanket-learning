__kernel void divide(__global float * x, __global float * y, int p)
{
   int row = get_global_id(0);
   x[row] /= y[row/p];
}