__kernel void adam(__global float * param, __global float * dparam, __global float * m, __global float * v, __global const float * t_)
{
   int row = get_global_id(0);
   float mt, vt, g, beta1, beta2;
   g = dparam[row];
   beta1 = t_[3];
   beta2 = t_[4];
   mt = beta1 * m[row] + (1.0 - beta1) * g;
   vt = beta2 * v[row] + (1.0 - beta2) * g * g;
   m[row] = mt;
   v[row] = vt;
   param[row] -= t_[1] * mt / (sqrt(vt) + t_[5]);
   dparam[row] = 0;
}