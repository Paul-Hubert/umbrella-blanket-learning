__kernel void sgd(__global float * param, __global float * dparam, float learn_rate)
{
   int row = get_global_id(0);
   param[row] -= dparam[row] * learn_rate;
   dparam[row] = 0;
}