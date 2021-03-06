package burlap.behavior.stochasticgame.agents.mavf;

import java.util.Map;

import burlap.behavior.singleagent.ValueFunctionInitialization;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.stochasticgame.mavaluefunction.MAValueFunctionPlanner;
import burlap.behavior.stochasticgame.mavaluefunction.SGBackupOperator;
import burlap.behavior.stochasticgame.mavaluefunction.vfplanners.MAValueIteration;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.stochasticgames.AgentType;
import burlap.oomdp.stochasticgames.JointActionModel;
import burlap.oomdp.stochasticgames.JointReward;
import burlap.oomdp.stochasticgames.SGDomain;

/**
 * An interface for generating multi agent value function planners ({@link MAValueFunctionPlanner} objects). This is useful for the
 * {@link MAVFPlanAgentFactory} since it allows planning objects to be delployed with the generation of agents. A factory that returns
 * a constant object reference is supplied as well as one for generating new value iteration instances.
 * 
 * @author James MacGlashan
 *
 */
public interface MAVFPlannerFactory {
	
	/**
	 * returns an {@link MAValueFunctionPlanner} reference to use for planning.
	 * @return an {@link MAValueFunctionPlanner} reference to use for planning.
	 */
	public MAValueFunctionPlanner getPlannerInstance();
	
	
	
	/**
	 * {@link MAValueFunctionPlanner} factory that always returns the same object instance, unless the reference is chaned with a mutator.
	 * @author James MacGlashan
	 *
	 */
	public static class ConstantMAVFPlannerFactory implements MAVFPlannerFactory{

		protected MAValueFunctionPlanner	plannerReferece;
		
		
		/**
		 * Initializes with a given planner reference.
		 * @param plannerRefernece the planner reference to return
		 */
		public ConstantMAVFPlannerFactory(MAValueFunctionPlanner plannerRefernece){
			this.plannerReferece = plannerRefernece;
		}
		
		/**
		 * Changes the planner reference
		 * @param plannerReference the planner reference to return
		 */
		public void setPlannerReference(MAValueFunctionPlanner plannerReference){
			this.plannerReferece = plannerReference;
		}
		
		@Override
		public MAValueFunctionPlanner getPlannerInstance() {
			return this.plannerReferece;
		}
		
	}
	
	
	
	
	/**
	 * Factory for generating multi-agent value iteration planners ({@link MAValueIteration}).
	 * @author James MacGlashan
	 *
	 */
	public static class MAVIPlannerFactory implements MAVFPlannerFactory{

		/**
		 * The domain in which planning is to be performed
		 */
		protected SGDomain						domain;
		
		/**
		 * The agent definitions for which planning is performed.
		 */
		protected Map<String, AgentType>		agentDefinitions = null;
		
		/**
		 * The joint action model to use in planning.
		 */
		protected JointActionModel				jointActionModel;
		
		/**
		 * The joint reward function
		 */
		protected JointReward					jointReward;
		
		/**
		 * The state terminal function.
		 */
		protected TerminalFunction				terminalFunction;
		
		/**
		 * The discount factor in [0, 1]
		 */
		protected double						discount;
		
		/**
		 * The state hashing factory used to query the value function for individual states
		 */
		protected StateHashFactory				hashingFactory;
		
		/**
		 * The Q-value initialization function to use.
		 */
		protected ValueFunctionInitialization	qInit;
		
		/**
		 * The backup operating defining the solution concept to use.
		 */
		protected SGBackupOperator				backupOperator;
		
		
		/**
		 * The threshold that will cause VI to terminate when the max change in Q-value for is less than it
		 */
		protected double 						maxDelta;
		
		/**
		 * The maximum allowable number of iterations until VI termination
		 */
		protected int 							maxIterations;
		
		
		/**
		 * Initializes.
		 * @param domain the domain in which to perform planing
		 * @param jointActionModel the joint action model
		 * @param jointReward the joint reward function
		 * @param terminalFunction the terminal state function
		 * @param discount the discount
		 * @param hashingFactory the hashing factory to use for storing states
		 * @param qInit the default Q-value to initialize all values to
		 * @param backupOperator the backup operator that defines the solution concept being solved
		 * @param maxDelta the threshold that causes VI to terminate when the max Q-value change is less than it
		 * @param maxIterations the maximum number of iterations allowed
		 */
		public MAVIPlannerFactory(SGDomain domain, JointActionModel jointActionModel, JointReward jointReward, TerminalFunction terminalFunction, 
				double discount, StateHashFactory hashingFactory, double qInit, SGBackupOperator backupOperator, double maxDelta, int maxIterations){
			
			this.domain = domain;
			this.jointActionModel = jointActionModel;
			this.jointReward = jointReward;
			this.terminalFunction = terminalFunction;
			this.discount = discount;
			this.hashingFactory = hashingFactory;
			this.qInit = new ValueFunctionInitialization.ConstantValueFunctionInitialization(qInit);
			this.backupOperator = backupOperator;this.maxDelta = maxDelta;
			this.maxIterations = maxIterations;
			
		}
		
		
		/**
		 * Initializes.
		 * @param domain the domain in which to perform planing
		 * @param jointActionModel the joint action model
		 * @param jointReward the joint reward function
		 * @param terminalFunction the terminal state function
		 * @param discount the discount
		 * @param hashingFactory the hashing factory to use for storing states
		 * @param qInit the q-value initialization function to use.
		 * @param backupOperator the backup operator that defines the solution concept being solved
		 * @param maxDelta the threshold that causes VI to terminate when the max Q-value change is less than it
		 * @param maxIterations the maximum number of iterations allowed
		 */
		public MAVIPlannerFactory(SGDomain domain, JointActionModel jointActionModel, JointReward jointReward, TerminalFunction terminalFunction, 
				double discount, StateHashFactory hashingFactory, ValueFunctionInitialization qInit, SGBackupOperator backupOperator, double maxDelta, int maxIterations){
			
			this.domain = domain;
			this.jointActionModel = jointActionModel;
			this.jointReward = jointReward;
			this.terminalFunction = terminalFunction;
			this.discount = discount;
			this.hashingFactory = hashingFactory;
			this.qInit = qInit;
			this.backupOperator = backupOperator;
			this.maxDelta = maxDelta;
			this.maxIterations = maxIterations;
			
		}
		
		
		/**
		 * Initializes.
		 * @param domain the domain in which to perform planing
		 * @param jointActionModel the joint action model
		 * @param jointReward the joint reward function
		 * @param terminalFunction the terminal state function
		 * @param discount the discount
		 * @param hashingFactory the hashing factory to use for storing states
		 * @param qInit the q-value initialization function to use.
		 * @param backupOperator the backup operator that defines the solution concept being solved
		 * @param maxDelta the threshold that causes VI to terminate when the max Q-value change is less than it
		 * @param maxIterations the maximum number of iterations allowed
		 */
		public MAVIPlannerFactory(SGDomain domain, Map<String, AgentType> agentDefinitions, JointActionModel jointActionModel, JointReward jointReward, TerminalFunction terminalFunction, 
				double discount, StateHashFactory hashingFactory, ValueFunctionInitialization qInit, SGBackupOperator backupOperator, double maxDelta, int maxIterations){
			
			this.domain = domain;
			this.agentDefinitions = agentDefinitions;
			this.jointActionModel = jointActionModel;
			this.jointReward = jointReward;
			this.terminalFunction = terminalFunction;
			this.discount = discount;
			this.hashingFactory = hashingFactory;
			this.qInit = qInit;
			this.backupOperator = backupOperator;
			this.maxDelta = maxDelta;
			this.maxIterations = maxIterations;
			
		}
		
		
		
		
		@Override
		public MAValueFunctionPlanner getPlannerInstance() {
			return new MAValueIteration(domain, agentDefinitions, jointActionModel, jointReward, terminalFunction, discount, hashingFactory, qInit, backupOperator, maxDelta, maxIterations);
		}
		
		
	}
	
	
	
	
}
