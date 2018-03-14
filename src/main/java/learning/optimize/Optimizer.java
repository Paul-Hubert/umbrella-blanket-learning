package learning.optimize;

import learning.tensor.*;

public abstract class Optimizer {
   
   public abstract void prepare();
   public abstract void optimize(Tensor param, Tensor dparam);
   
}