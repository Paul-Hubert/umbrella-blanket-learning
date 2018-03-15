package learning.operation;

import learning.node.Node;
import learning.tensor.Tensor;

public class Operation {
   
   private Node current;
   public boolean calculate = true;
   
   public void prepare() {
      
   }
   
   public void operate(Node n, Tensor t) {
      current = n;
   }
   
   public Node getLast() {
      return current;
   }
   
   
}