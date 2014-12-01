package burlap.testcode;

import java.util.*;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.StateConditionTest;
import burlap.behavior.singleagent.planning.commonpolicies.BoltzmannQPolicy;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.SDPlannerPolicy;
import burlap.behavior.singleagent.planning.deterministic.TFGoalCondition;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.statehashing.NameDependentStateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.debugtools.MyTimer;
import burlap.domain.singleagent.blocksworld.BlocksWorld;
import burlap.domain.singleagent.blocksworld.BlocksWorldVisualizer;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.auxiliary.common.UniversalStateParser;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.GroundedProp;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.visualizer.Visualizer;

public class BlockWorldTest {

    BlocksWorld							bwd;
    Domain								domain;
    StateParser 						sp;
    RewardFunction 						rf;
    TerminalFunction					tf;
    StateConditionTest					goalCondition;
    State 								initialState;
    NameDependentStateHashFactory		hashingFactory;


    /**
     * @param args
     */
    public static void main(String[] args) {
        BlockWorldTest rpt = new BlockWorldTest();
        String outputPath = "output"; //directory to record results

        System.out.println("\nValue Iteration (Gamma=0.7)");
        System.out.println("----------------------------\n");
        rpt.ValueIterationExample(outputPath, 0.7);

        System.out.println("\nValue Iteration (Gamma=0.8)");
        System.out.println("----------------------------\n");
        rpt.ValueIterationExample(outputPath, 0.8);

        System.out.println("\nValue Iteration (Gamma=0.9)");
        System.out.println("----------------------------\n");
        rpt.ValueIterationExample(outputPath, 0.9);

        System.out.println("\nValue Iteration (Gamma=0.99)");
        System.out.println("----------------------------\n");
        rpt.ValueIterationExample(outputPath, 0.99);

        System.out.println("\nPolicy Iteration (Gamma=0.7)");
        System.out.println("----------------------------\n");
        rpt.PolicyIterationExample(outputPath, 0.7);

        System.out.println("\nPolicy Iteration (Gamma=0.8)");
        System.out.println("----------------------------\n");
        rpt.PolicyIterationExample(outputPath, 0.8);

        System.out.println("\nPolicy Iteration (Gamma=0.9)");
        System.out.println("----------------------------\n");
        rpt.PolicyIterationExample(outputPath, 0.9);

        System.out.println("\nPolicy Iteration (Gamma=0.99)");
        System.out.println("----------------------------\n");
        rpt.PolicyIterationExample(outputPath, 0.99);

        System.out.println("\nQ-Learning Iteration (Gamma=0.7)");
        System.out.println("----------------------------\n");
        rpt.QLearningExample(outputPath, 0.7);

        System.out.println("\nQ-Learning Iteration (Gamma=0.8)");
        System.out.println("----------------------------\n");
        rpt.QLearningExample(outputPath, 0.8);

        System.out.println("\nQ-Learning Iteration (Gamma=0.9)");
        System.out.println("----------------------------\n");
        rpt.QLearningExample(outputPath, 0.9);

        System.out.println("\nQ-Learning Iteration (Gamma=0.99)");
        System.out.println("----------------------------\n");
        rpt.QLearningExample(outputPath, 0.99);

        //rpt.PolicyIterationExample(outputPath, 0.99);
        //rpt.QLearningExample(outputPath, 0.99);

        rpt.visualize(outputPath);


        rpt.efficiencyTest();

    }



    public BlockWorldTest(){
        bwd = new BlocksWorld();
        domain = bwd.generateDomain();
        sp = new UniversalStateParser(domain);
        rf = new UniformCostRF();
        tf = new StackTerminal();
        goalCondition = new TFGoalCondition(tf);

        initialState = BlocksWorld.getNewState(domain, 6);

        hashingFactory = new NameDependentStateHashFactory();


    }

    public void visualize(String outputPath){
        Visualizer v = BlocksWorldVisualizer.getVisualizer();
        EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v, domain, sp, outputPath);
    }


    public void BFSExample(String outputPath){

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        //BFS ignores reward; it just searches for a goal condition satisfying state
        DeterministicPlanner planner = new BFS(domain, goalCondition, hashingFactory);
        planner.planFromState(initialState);

        //capture the computed plan in a partial policy
        Policy p = new SDPlannerPolicy(planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult", sp);

    }

    public void ValueIterationExample(String outputPath, double gamma){

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        double start = System.nanoTime(), end = 0, runTime = 0;

        //Value iteration computing for discount=0.99 with stopping criteria either being a maximum change in value less then 0.001 or 100 passes over the state space (which ever comes first)
        OOMDPPlanner planner = new ValueIteration(domain, rf, tf, gamma, hashingFactory, 0.001, 100);
        planner.planFromState(initialState);

        end = System.nanoTime();
        runTime = end - start;

        runTime = runTime/10e9;
        System.out.println("Planner VI time: " + runTime);

        start = System.nanoTime();
        //create a Q-greedy policy from the planner
        Policy p = new GreedyQPolicy((QComputablePlanner)planner);

        //record the plan results to a file
        p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "planResult", sp);
        end = System.nanoTime();
        runTime = end - start;

        runTime = runTime/10e9;
        System.out.println("VI greedy evaluation time: " + runTime);

    }

    public void PolicyIterationExample(String outputPath, double gamma) {

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        double start = System.nanoTime(), end = 0, evalTime = 0;
        OOMDPPlanner planner = new PolicyIteration(domain, rf, tf, gamma, hashingFactory, 0.001, 100, 1000);
        planner.planFromState(initialState);
        end = System.nanoTime();
        evalTime = end - start;

        evalTime /= Math.pow(10, 9);
        System.out.println("Planner PI time: " + evalTime);
        //create a Q-greedy policy from the planner
        Policy p = new GreedyQPolicy((QComputablePlanner)planner);
        start = System.nanoTime();
        //record the plan results to a file
        EpisodeAnalysis analysis = p.evaluateBehavior(initialState, rf, tf);
        end = System.nanoTime();
        evalTime = end - start;
        evalTime /= Math.pow(10, 9);
        LinkedList<State> uniqueStates = new LinkedList<State>();
        System.out.println("Number of states to termination PI:" + analysis.stateSequence.size());
        for(State s : analysis.stateSequence)
        {
            if(!uniqueStates.contains(s))
                uniqueStates.add(s);
        }
        System.out.println("Unique states in policy PI: "+ uniqueStates.size());
        System.out.println("PI time: " + evalTime);
        System.out.println("Discount reward PI: " + analysis.getDiscountedReturn(gamma));
        double totalReward = 0;

        for( double r : analysis.rewardSequence)
        {
            totalReward += r;
        }
        System.out.println("Total reward PI: " + totalReward);
        analysis.writeToFile(outputPath + "planResult", sp);
        //this.visualizePolicy((QComputablePlanner) planner, p);
    }

    public void QLearningExample(String outputPath, double gamma){

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }

        double start = System.nanoTime(), end = 0, evalTime = 0;
        //creating the learning algorithm object; discount= 0.99; initialQ=0.0; learning rate=0.9
        LearningAgent agent = new QLearning(domain, rf, tf, gamma, hashingFactory, 0., 0.9);

        //run learning for 100 episodes
        for(int i = 0; i < 200; i++){
            EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState); //run learning episode
            ea.writeToFile(String.format("%se%03d", outputPath, i), sp); //record episode to a file
            System.out.println(i + ": " + ea.numTimeSteps()); //print the performance of this episode
        }

        end = System.nanoTime();
        evalTime = end - start;
        evalTime /= Math.pow(10, 9);
        System.out.println("Q Learning time: " + evalTime);
        start = System.nanoTime();

//        Policy p = new GreedyQPolicy((QComputablePlanner)((QLearning)agent));
        Policy p = new BoltzmannQPolicy((QComputablePlanner)((QLearning)agent), 10);

        EpisodeAnalysis analysis = p.evaluateBehavior(initialState, rf, tf);
        end = System.nanoTime();
        evalTime = end - start;
        evalTime /= Math.pow(10, 9);
        System.out.println("Q-Learning Eval time: " + evalTime);
        LinkedList<State> uniqueStates = new LinkedList<State>();
        System.out.println("Number of states to termination Q-Learning:" + analysis.stateSequence.size());
        for(State s : analysis.stateSequence)
        {
            if(!uniqueStates.contains(s))
                uniqueStates.add(s);
        }
        System.out.println("Unique states in Q-Learning: "+ uniqueStates.size());
        System.out.println("Q time: " + evalTime);
        System.out.println("Discount reward Q: " + analysis.getDiscountedReturn(0.99));
        double totalReward = 0;

        for( double r : analysis.rewardSequence)
        {
            totalReward += r;
        }
        System.out.println("Total reward Q-Learning: " + totalReward);
        analysis.writeToFile(outputPath + "planResult", sp);

    }


    public class StackTerminal implements TerminalFunction{

        List<GroundedProp> gps;

        public StackTerminal(){
            gps = new ArrayList<GroundedProp>();
            gps.add(new GroundedProp(domain.getPropFunction(BlocksWorld.PFONBLOCK), new String[]{"block0", "block1"}));
            gps.add(new GroundedProp(domain.getPropFunction(BlocksWorld.PFONBLOCK), new String[]{"block2", "block0"}));
            gps.add(new GroundedProp(domain.getPropFunction(BlocksWorld.PFONBLOCK), new String[]{"block3", "block4"}));
            gps.add(new GroundedProp(domain.getPropFunction(BlocksWorld.PFONBLOCK), new String[]{"block4", "block5"}));
        }

        @Override
        public boolean isTerminal(State s) {
            for(GroundedProp gp : gps){
                if(!gp.isTrue(s)){
                    return false;
                }
            }
            return true;
        }


    }



    public void efficiencyTest(){

        Set <StateHashTuple> reachability;
        MyTimer timer = new MyTimer();

        System.out.println("Starting");

        timer.start();
        reachability = StateReachability.getReachableHashedStates(initialState, (SADomain)domain, hashingFactory);
        timer.stop();

        System.out.println("Reachability Time: " +  timer.getTime() + "; n states: " + reachability.size());

        Set <StateHashTuple> copiedSet = new HashSet<StateHashTuple>(reachability.size());
        timer.start();
        for(StateHashTuple sh : reachability){
            StateHashTuple nsh = this.hashingFactory.hashState(sh.s.copy());
            copiedSet.add(nsh);
        }
        timer.stop();
        System.out.println("Set Copy Time: " + timer.getTime() + "; n states: " + copiedSet.size());

        List<State> flatList = new ArrayList<State>();
        timer.start();
        for(StateHashTuple sh : reachability){
            State ns = sh.s.copy();
            flatList.add(ns);
        }
        timer.stop();
        System.out.println("Flat List Time: " + timer.getTime() + "; n states: " + flatList.size());

        timer.start();
        for(StateHashTuple sh : reachability){
            StateHashTuple nsh = this.hashingFactory.hashState(sh.s.copy());
            nsh.hashCode(); //forces hashCode value compute
        }
        timer.stop();
        System.out.println("Hash Time: " + timer.getTime() + "; n states: " + copiedSet.size());

        Set <Integer> hashCodes = new HashSet<Integer>();
        timer.start();
        for(StateHashTuple sh : reachability){
            StateHashTuple nsh = this.hashingFactory.hashState(sh.s.copy());
            int hc = nsh.hashCode(); //forces hashCode value compute
            hashCodes.add(hc);
        }
        timer.stop();
        System.out.println("Hash Code Index Time: " + timer.getTime() + "; n hash codes: " + hashCodes.size());

		/*for(int hc : hashCodes){
			System.out.println(hc);
		}*/


    }


    public void efficiencyTest2(){

        Set <StateHashTuple> reachability;
        MyTimer timer = new MyTimer();

        System.out.println("Starting");

        timer.start();
        reachability = StateReachability.getReachableHashedStates(initialState, (SADomain)domain, hashingFactory);
        timer.stop();

        System.out.println("Time: " +  timer.getTime() + "; n states: " + reachability.size());


    }


    public void efficiencyTest1(){

        int n = 24306;
        MyTimer timer = new MyTimer();


        List<StateHashTuple> shList = new ArrayList<StateHashTuple>();
        timer.start();
        for(int i = 0; i < n; i++){
            State s = initialState.copy();
            StateHashTuple sh = this.hashingFactory.hashState(s);
            sh.hashCode();
            shList.add(sh);
        }

        timer.stop();
        System.out.println("List time: " + timer.getTime());
        System.out.println(shList.size());

        Set<StateHashTuple> shSet = new HashSet<StateHashTuple>();
        timer.start();
        for(int i = 0; i < n; i++){
            State s = initialState.copy();
            StateHashTuple sh = this.hashingFactory.hashState(s);
            shSet.add(sh);
        }

        timer.stop();
        System.out.println("Set time: " + timer.getTime());
        System.out.println(shSet.size());

    }


}
