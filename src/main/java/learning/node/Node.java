package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;
import learning.Network;

public class Node {
   
   protected Network net;
   
   protected Node next, before;
   
   protected Tensor input, in, output, dO;
   
   protected int[] outputSize;
   
   protected boolean ready = false;
   protected String name;
   
   public Node() {
      setNetwork();
   }
   
   public void forwardProp() {
      output = in;
   }
   
   public void backwardProp() {
      input = dO;
   }
   
   Tensor forwards() {
      in = before.forwards();
      net.operate(this);
      return output;
   }
   
   void backwards(Tensor t) {
      dO = t;
      net.operate(this);
      before.backwards(input);
   }
   
   protected void attachNext(Node n) {
      next = n;
   }
   
   protected void attachBefore(Node n) {
      before = n;
      setOutputSize();
      prepare();
   }
   
   public void optimize(Optimizer opt) {
      //overriden for nodes with parameters;
   }
   
   public int[] getOutputSize() {
      return outputSize;
   }
   
   protected void setOutputSize() {
      outputSize = before.getOutputSize(); // is automatically called if attachBefore is not overriden
   }
   
   protected void prepare() {
     //overriden for Tensor creation and kernel compiling
   }
   
   public final Tensor forwardOp() {
      return this.forwards();
   }
   
   public final void backwardOp(Tensor t) {
      this.backwards(t);
   }
   
   public final Tensor getOutput() {
      return output;
   }
   
   public final Tensor getGradient() {
      return input;
   }
   
   public final void setNetwork() {
      net = Network.getCurrent();
      net.add(this);
   }
   
   public final Node chain(Node n) {
      this.attachNext(n);
      n.attachBefore(this);
      return n;
   }
   
   public final Node name(String s) {
      name = s;
      return this;
   }
   
}