package learning.operation;

import learning.node.Node;
import learning.tensor.Tensor;

public abstract class Operation {
   
   private Node current;
   
   public abstract void prepare();
   
   public final void operate(Node n) {
      operation(n);
      current = n;
   }
   
   protected abstract void operation(Node n);
   
   public Node getLast() {
      return current;
   }
   
   
}