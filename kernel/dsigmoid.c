__kernel void dsigmoid(__global const float * x, __global const float * sigmoid, __global float * y)
{
   int row = get_global_id(0);
   y[row] = sigmoid[row] * (1.0-sigmoid[row]) * x[row];
}