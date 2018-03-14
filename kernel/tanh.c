__kernel void tanh_(__global const float * x, __global float * y, __global float * t)
{
   int row = get_global_id(0);
   float xi = x[i];
   float expp = exp(xi), expm = exp(-xi);
   float temp = (expp - expm)/(expp + expm);
   y[row] = temp;
   t[row] = temp;
}