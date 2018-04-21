package learning.operation;

import learning.node.Node;
import learning.tensor.Tensor;
import learning.optimize.Optimizer;

public class OptimizeOperation extends Operation {
   
   private Optimizer opt;
   
   public void prepare() {
      opt.prepare();
   }
   
   protected void operation(Node n) {
      n.optimize(opt);
   }
   
   public void setOptimizer(Optimizer opt) {
      this.opt = opt;
   }
   
   
}