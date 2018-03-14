__kernel void sigmoid(__global const float * x, __global float * y, __global float * t)
{
   int row = get_global_id(0);
   float temp =  1/(1 + exp(-x[row]));
   y[row] = temp;
   t[row] = temp;
}