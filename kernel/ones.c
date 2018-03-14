__kernel void ones(__global float * y)
{
   int col = get_global_id(0);
   y[col] = 1.0;
}