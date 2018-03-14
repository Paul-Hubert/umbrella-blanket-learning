package learning.operation;

import learning.node.Node;
import learning.tensor.Tensor;
import learning.optimize.Optimizer;

public class OptimizeOperation extends Operation {
   
   private Optimizer opt;
   
   public OptimizeOperation() {
      calculate = false;
   }
   
   public void prepare() {
      
   }
   
   public void operate(Node n, Tensor t) {
      n.optimize(opt);
   }
   
   public void setOptimizer(Optimizer opt) {
      this.opt = opt;
   }
   
   
}