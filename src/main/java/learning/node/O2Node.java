package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class O2Node extends Node {
   
   protected Node next2;
   
   protected Tensor output2, dO2;
   
   protected int last = -1;
   
   public void forwardProp() {
      output = in;
      output2 = in;
   }
   
   public void backwardProp() {
      input = dO;
   }
   
   Tensor forwards() {
      if(net.getOpID() != last) {
         in = before.forwards();
         net.operate(this);
         last = net.getOpID();
      }
      return output;
   }
   
   void backwards(Tensor t) {
      if(net.getLast() == next) dO = t;
      if(net.getLast() == next2) dO2 = t;
      if(dO != null && dO2 != null) {
         net.operate(this);
         before.backwards(input);
         dO = null;
         dO2 = null;
      }
   }
   
   protected void attachNext(Node n) {
      if(next==null) {
         next = n;
      } else {
         next2 = n;
      }
   }
   
}