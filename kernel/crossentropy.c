__kernel void crossentropy(__global const float * x, __global float * y, __global const float * label, int n)
{
   int row = get_global_id(0);
   y[row] = -log((label[row/n]!=row%n ? (float) (1)-x[row] : x[row])) / (float)(n);
}