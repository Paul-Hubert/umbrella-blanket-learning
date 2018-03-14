// Reduce M = get_global_size(0) rows of P values in matrix Y.
// Stores the result in first column of Y.
__kernel void reduce_rows(__global const float * x, __global float * y, int p)
{
  int row = get_global_id(0);
  int size = get_global_size(0);
  float sum = (float) 0;
  for (int col=0;col<p;col++) sum += x[p*row + col];
  y[row] = sum;
}