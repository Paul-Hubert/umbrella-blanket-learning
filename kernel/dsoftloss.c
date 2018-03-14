__kernel void dsoftloss(__global const float * label, __global const float * soft, __global const float * x, __global float * y, int row_size)
{
   int row = get_global_id(0);
   y[row] = (soft[row] - (float) (label[row/row_size]==row%row_size ? 1 : 0)) * x[row/row_size];
}