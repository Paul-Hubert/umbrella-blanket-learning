__kernel void dconcatenate(__global const float * x,__global float * y1, __global float * y2, int offset)
{
   int col = get_global_id(0);
   if(col<offset) {
      y1[col] = x[col];
   } else {
      y2[col-offset] = x[col];
   }
}