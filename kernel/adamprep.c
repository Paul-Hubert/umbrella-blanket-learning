__kernel void adamprep(__global float * t_)
{
   float t = t_[0] + 1;
   t_[0] = t;
   t_[1] = t_[2] * sqrt(1.0 - pow(t_[4],t)) / (1.0 - pow(t_[3],t));
}