__kernel void exponential(__global const float * x, __global float * y)
{
   int row = get_global_id(0);
   y[row] = exp(x[row] + 0.00001);
}