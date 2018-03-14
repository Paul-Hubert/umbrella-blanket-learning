__kernel void dlinear(__global const float * w, __global float * dw, __global float * db, __global const float * x, __global const float * dy, __global float * dx, int mrow)
{
   int col = get_global_id(0), ncol = get_global_size(0);
   float _dx = 0, _x = x[col];
   for(int row = 0; row<mrow; row++) {
      float _dy = dy[row];
      _dx += w[row*ncol+col]*_dy;
      dw[row*ncol+col] += _x*_dy;
   }
   dx[col] = _dx;
}