package burlap.testcode;

import java.awt.Color;
import java.util.LinkedList;
import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D.PolicyGlyphRenderStyle;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.StateConditionTest;
import burlap.behavior.singleagent.planning.commonpolicies.BoltzmannQPolicy;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.TFGoalCondition;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.SinglePFTF;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.singleagent.common.VisualActionObserver;

public class Maze {
    private static final boolean POLICY_VIS_ON = true;
    // The probability that our agent moves in the correct direction
    private static final double probSucceed = 0.8;

    GridWorldDomain				gwdg;
    Domain						domain;
    StateParser					sp;
    RewardFunction				rf;
    TerminalFunction			tf;
    StateConditionTest			goalCondition;
    State						initialState;
    DiscreteStateHashFactory	hashingFactory;

    int myMap[][] = {
            {1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,0,1,0,1,0,1,0,0,0,0,0,1},
            {1,0,1,0,0,0,1,0,1,1,1,0,1},
            {1,0,0,0,1,1,1,0,0,0,0,0,1},
            {1,0,1,0,0,0,0,0,1,1,1,0,1},
            {1,0,1,0,1,1,1,0,1,0,0,0,1},
            {1,0,1,0,1,0,0,0,1,1,1,0,1},
            {1,0,1,0,1,1,1,0,1,0,1,0,1},
            {1,0,0,0,0,0,0,0,0,0,1,0,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1}
    };

    public Maze(){
        // Setup our Grid World Domain
        gwdg = new GridWorldDomain(13, 13);
        gwdg.setProbSucceedTransitionDynamics(probSucceed);
        gwdg.setMap(myMap);
        domain = gwdg.generateDomain();

        //Setup initial state of our agent and reward location
        initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
        GridWorldDomain.setAgent(initialState, 1, 1);
        GridWorldDomain.setLocation(initialState, 0, 1, 8);

        sp = new GridWorldStateParser(domain);
        rf = new UniformCostRF();
        tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));
        goalCondition = new TFGoalCondition(tf);
        hashingFactory = new DiscreteStateHashFactory();
        hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT,
                domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);
//        VisualActionObserver observer = new VisualActionObserver(domain,
//        		GridWorldVisualizer.getVisualizer(gwdg.getMap()));
//        ((SADomain)this.domain).setActionObserverForAllAction(observer);
//        observer.initGUI();
    }

    public static void main(String[] args) {

        Maze testMaze = new Maze();
        String outputFolder = "output/"; //directory to record results

        System.out.println("\nValue Iteration (Gamma=0.99)");
        System.out.println("----------------------------\n");
        testMaze.testValueIteration(outputFolder, 0.99);

        System.out.println("\nPolicy Iteration (Gamma=0.99)");
        System.out.println("----------------------------\n");
        testMaze.testPolicyIteration(outputFolder, 0.99);

        System.out.println("\nQ-Learning (Gamma=0.99)");
        System.out.println("----------------------------\n");
        testMaze.testQLearning(outputFolder, 0.99);


        System.out.println("\nValue Iteration (Gamma=0.9)");
        System.out.println("----------------------------\n");
        testMaze.testValueIteration(outputFolder, .9);

        System.out.println("\nPolicy Iteration (Gamma=0.9)");
        System.out.println("----------------------------\n");
        testMaze.testPolicyIteration(outputFolder, .9);

        System.out.println("\nQ-Learning (Gamma=0.9)");
        System.out.println("----------------------------\n");
        testMaze.testQLearning(outputFolder, 0.9);
//
        System.out.println("\nValue Iteration (Gamma=0.8)");
        System.out.println("----------------------------\n");
        testMaze.testValueIteration(outputFolder, 0.8);

        System.out.println("\nPolicy Iteration (Gamma=0.8)");
        System.out.println("----------------------------\n");
        testMaze.testPolicyIteration(outputFolder, 0.8);

        System.out.println("\nQ-Learning (Gamma=0.8)");
        System.out.println("----------------------------\n");
        testMaze.testQLearning(outputFolder, 0.8);
//
        System.out.println("\nValue Iteration (Gamma=0.7)");
        System.out.println("----------------------------\n");
        testMaze.testValueIteration(outputFolder, 0.7);

        System.out.println("\nPolicy Iteration (Gamma=0.7)");
        System.out.println("----------------------------\n");
        testMaze.testPolicyIteration(outputFolder, 0.7);

        System.out.println("\nQ-Learning (Gamma=0.7)");
        System.out.println("----------------------------\n");
        testMaze.testQLearning(outputFolder, 0.7);
//
        System.out.println("\nValue Iteration (Gamma=0.6)");
        System.out.println("----------------------------\n");
        testMaze.testValueIteration(outputFolder, 0.6);

        System.out.println("\nPolicy Iteration (Gamma=0.6)");
        System.out.println("----------------------------\n");
        testMaze.testPolicyIteration(outputFolder, 0.6);

    }

    public void visualizePolicy(QComputablePlanner planner, Policy p){
        List <State> allStates = StateReachability.getReachableStates(initialState,
                (SADomain)domain, hashingFactory);
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
                GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);

        PolicyGlyphPainter2D spp = ArrowActionGlyph.getNSEWPolicyGlyphPainter(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
                GridWorldDomain.ATTY, GridWorldDomain.ACTIONNORTH,
                GridWorldDomain.ACTIONSOUTH, GridWorldDomain.ACTIONEAST, GridWorldDomain.ACTIONWEST);

        spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);

        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, planner);
        gui.setBgColor(Color.WHITE);
        gui.setSpp(spp);
        gui.setPolicy(p);
        gui.initGUI();
    }

    public void qLearningPlotter(){

        /**
         * Create factory for Q-learning agent
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            @Override
            public String getAgentName() {
                return "Q-learning";
            }

            @Override
            public LearningAgent generateAgent() {
                return new QLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);
            }
        };

        StateGenerator sg = new ConstantStateGenerator(this.initialState);

        // define experiment
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter((SADomain)this.domain, rf, sg,
                10, 100, qLearningFactory);

        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOSTRECENTANDAVERAGE,
                PerformanceMetric.CUMULATIVESTEPSPEREPISODE,
                PerformanceMetric.AVERAGEEPISODEREWARD);

        // start experiment
        exp.startExperiment();

        exp.writeStepAndEpisodeDataToCSV("qLearningOutput");
    }

    public void policyPlotter(){

    }

    public void testValueIteration(String directory, double gamma){

        if(!directory.endsWith("/")){
            directory = directory + "/";
        }

        double start = System.nanoTime(), end = 0, runTime = 0;

        OOMDPPlanner planner = new ValueIteration(domain, rf, tf, gamma, hashingFactory, 0.001, 200);
        planner.planFromState(initialState);

        end = System.nanoTime();
        runTime = end - start;

        runTime = runTime/10e9;
        System.out.println("Planner VI time: " + runTime);
        //create a Q-greedy policy from the planner
        start = System.nanoTime();
        Policy p = new GreedyQPolicy((QComputablePlanner)planner);

        //record the plan results to a file
        EpisodeAnalysis analysis = p.evaluateBehavior(initialState, rf, tf);
        end = System.nanoTime();
        runTime = end - start;

        runTime = runTime/10e9;
        System.out.println("VI greedy evaluation time: " + runTime);
        LinkedList<State> uniqueStates = new LinkedList<State>();
        System.out.println("Number of states to termination VI:" + analysis.stateSequence.size());
        for(State s : analysis.stateSequence)
        {
            if(!uniqueStates.contains(s))
                uniqueStates.add(s);
        }
        System.out.println("Unique states in policy VI: "+ uniqueStates.size());
        System.out.println("Discount reward VI: " + analysis.getDiscountedReturn(0.99));
        double totalReward = 0;

        for( double r : analysis.rewardSequence)
        {
            totalReward += r;
        }
        System.out.println("Total reward VI: " + totalReward);
        analysis.writeToFile(directory + "planResult", sp);
        if(POLICY_VIS_ON)
            this.visualizePolicy((QComputablePlanner) planner, p);
    }

    public void testPolicyIteration(String outputPath, double gamma) {

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
        if(POLICY_VIS_ON)
            this.visualizePolicy((QComputablePlanner) planner, p);
    }

    public void testQLearning(String outputPath, double gamma){

        if(!outputPath.endsWith("/")){
            outputPath = outputPath + "/";
        }
        double start = System.nanoTime(), end = 0, evalTime = 0;

        LearningAgent agent = new QLearning(domain, rf, tf, gamma, hashingFactory, 0., 0.9);
        //run learning for 100 episodes
        for(int i = 0; i < 100; i++){
            agent.runLearningEpisodeFrom(initialState);
        }
        end = System.nanoTime();
        evalTime = end - start;
        evalTime /= Math.pow(10, 9);
        System.out.println("Q Learning time: " + evalTime);
        start = System.nanoTime();
        //Policy p = new GreedyQPolicy((QComputablePlanner)((QLearning)agent));
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
        if(POLICY_VIS_ON)
        {
            //this.visualizePolicy((QComputablePlanner) ((QLearning) agent), p);
            this.qLearningPlotter();
        }
    }
}