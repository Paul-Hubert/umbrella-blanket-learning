__kernel void zeros(__global float * y)
{
   int col = get_global_id(0);
   y[col] = 0.0;
}