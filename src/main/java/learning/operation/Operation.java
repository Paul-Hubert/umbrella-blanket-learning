package learning.operation;

import learning.node.Node;
import learning.tensor.Tensor;

public class Operation {
   
   public Node current;
   public boolean calculate = true;
   
   public void prepare() {
      
   }
   
   public void operate(Node n, Tensor t) {
      current = n;
   }
   
   
}