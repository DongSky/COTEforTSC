import java.io.File;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;
import java.util.TreeSet;
import java.util.function.DoubleToLongFunction;

import weka.core.Instances;
import weka.core.SystemInfo;

public class FlatCOTE{
    public static String cvDir="Result/cvAccs";
    public static String trainTestDir="Results/unifiedTrainTest";
    public static String instancesLoc="TSC Problems";
    public enum EnsembleType{
            Best,Equal,Prop,TopHalf,
    }
    public enum ClassifierType{
        Euclidean_1NN, DTW_R1_1NN, DTW_Rn_1NN, WDTW_1NN, DDTW_R1_1NN,
        DDTW_Rn_1NN, WDDTW_1NN, LCSS_1NN, ERP_1NN, MSM_1NN, TWE_1NN,
    
        Shapelet_1NN, Shapelet_BayesNet, Shapelet_NaiveBayes, Shapelet_J48,
        Shapelet_RanFor, Shapelet_RotFor, Shapelet_SVML, Shapelet_SVMQ,
    
        PS_OneNN, PS_BayesNet, PS_NaiveBayes, PS_J48,
        PS_RanFor, PS_RotFor, PS_SVML, PS_SVMQ,
    
        ACF_OneNN, ACF_BayesNet, ACF_NaiveBayes, ACF_J48,
        ACF_RanFor, ACF_RotFor, ACF_SVML, ACF_SVMQ,
    }
    
    private TreeSet<ClassifierType> Classifiers;
    private EnsembleType ensembleType;
    
    public FlatCOTE(){
        this.Classifiers=new TreeSet<>();
        this.ensembleType=EnsembleType.Prop;
    }
    //choose classifiers
    public void chooseAll(){
        this.Classifiers.addAll(Arrays.asList(ClassifierType.values()));
    }
    public void chooseShapelet(){
        this.Classifiers.add(ClassifierType.Shapelet_1NN);
        this.Classifiers.add(ClassifierType.Shapelet_NaiveBayes);
        this.Classifiers.add(ClassifierType.Shapelet_BayesNet);
        this.Classifiers.add(ClassifierType.Shapelet_J48);
        this.Classifiers.add(ClassifierType.Shapelet_RanFor);
        this.Classifiers.add(ClassifierType.Shapelet_RotFor);
        this.Classifiers.add(ClassifierType.Shapelet_SVML);
        this.Classifiers.add(ClassifierType.Shapelet_SVMQ);
    }
    public void chooseACF(){
        this.Classifiers.add(ClassifierType.ACF_BayesNet);
        this.Classifiers.add(ClassifierType.ACF_J48);
        this.Classifiers.add(ClassifierType.ACF_NaiveBayes);
        this.Classifiers.add(ClassifierType.ACF_OneNN);
        this.Classifiers.add(ClassifierType.ACF_RanFor);
        this.Classifiers.add(ClassifierType.ACF_RotFor);
        this.Classifiers.add(ClassifierType.ACF_SVML);
        this.Classifiers.add(ClassifierType.ACF_SVMQ);
    }
    public void chooseNN(){
        this.Classifiers.add(ClassifierType.Euclidean_1NN);
        this.Classifiers.add(ClassifierType.DTW_R1_1NN);
        this.Classifiers.add(ClassifierType.DTW_Rn_1NN);
        this.Classifiers.add(ClassifierType.ERP_1NN);
        this.Classifiers.add(ClassifierType.LCSS_1NN);
        this.Classifiers.add(ClassifierType.MSM_1NN);
        this.Classifiers.add(ClassifierType.TWE_1NN);
        this.Classifiers.add(ClassifierType.WDDTW_1NN);
        this.Classifiers.add(ClassifierType.WDTW_1NN);
        this.Classifiers.add(ClassifierType.DDTW_R1_1NN);
        this.Classifiers.add(ClassifierType.DDTW_Rn_1NN);
    }
    public void choosePS(){
        this.Classifiers.add(ClassifierType.PS_BayesNet);
        this.Classifiers.add(ClassifierType.PS_J48);
        this.Classifiers.add(ClassifierType.PS_NaiveBayes);
        this.Classifiers.add(ClassifierType.PS_OneNN);
        this.Classifiers.add(ClassifierType.PS_RanFor);
        this.Classifiers.add(ClassifierType.PS_RotFor);
        this.Classifiers.add(ClassifierType.PS_SVML);
        this.Classifiers.add(ClassifierType.PS_SVMQ);
    }
    //function to load data
    public static Instances loadData(String fileName){
        Instances data=null;
        try{
            FileReader reader=new FileReader(fileName);
            data=new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);
        }
        catch(Exception e){
            System.out.println("Error="+e+" in method loadData");
            e.printStackTrace();
        }
        return data;
    }
    //function to get cvAccuracy data
    public static double getCvAcc(ClassifierType classifierType,String dataSetName) throws FileNotFoundException{
        File CvFile=new File(cvDir+"/"+classifierType+"/cvAcc_"+classifierType+"_"+dataSetName+".txt");
        if(!CvFile.exists()){
            throw new FileNotFoundException("could not locale the cv file of "+classifierType+" and "+dataSetName);
        }
        Scanner s=new Scanner(CvFile);
        return Double.parseDouble(s.next().trim());
    }
    //function to get Train or Test Accuracy data
    public static double getTrainTestAcc(ClassifierType classifierType,String dataSetName) throws FileNotFoundException{
        File trainTestFile=new File(trainTestDir+"/"+classifierType+"/trainTest_"+classifierType+"_"+dataSetName+".txt");
        if(!trainTestFile.exists()){
            throw new FileNotFoundException("could not locale the train or testfile of "+classifierType+" and "+dataSetName);
        }
        Scanner s=new Scanner(trainTestFile);
        String[] parts=s.next().split("/");
        return Double.parseDouble(parts[0].trim())/Double.parseDouble(parts[1].trim());
    }
    //function to get Actual Class values
    public static double[] getActualClassValues(String dataSetName){
        Instances testInst=loadData(instancesLoc+"/"+dataSetName+"/"+dataSetName+"_TEST.arff");
        double[] classValues=new double[testInst.numInstances()];
        for(int i=0;i<testInst.numInstances();i++){
            classValues[i]=testInst.instance(i).classValue();
        }
        return classValues;
    }
    //some functions used to get Predictions
    //function to get test predictions
    public static double[] getTestPredictions(ClassifierType classifierType,String dataSetName) throws Exception{
        File trainTestResults=new File(trainTestDir+"/"+classifierType+"/trainTest_"+classifierType+"_"+dataSetName+".txt");
        if(!trainTestResults.exists()){
            throw new FileNotFoundException("TrainTest file not found: " + dataSetName + ", " + classifierType);
        }
        Scanner s=new Scanner(trainTestResults);
        s.useDelimiter("\n");
        String[] results=s.next().split("/");
        double[] predictions=new double[Integer.parseInt(results[1].trim())];
        int i=0;
        while(s.hasNext()){
            predictions[i++]=Double.parseDouble(s.next().split(",")[0].trim());
            if(i!=predictions.length){
                throw new Exception("Problem with the data: read in "+i+" instances, expected "+predictions.length);
            }
        }
        return predictions;
    }
    /*
    functions to print some data
     */
    //function to print Train Test Accuracy
    public static void printTrainTestAccuracy(String[] dataSets){
        System.out.println("DataSet ");
        for(int i=0;i<ClassifierType.values().length;i++){
            System.out.println(ClassifierType.values()[i]+",");
        }
        System.out.println("==========");
        ClassifierType classifier;
        String datasetName;
        for(int i=0;i<dataSets.length;i++){
            datasetName=dataSets[i];
            System.out.println(datasetName.replaceAll("normalised","")+",");
            for(int j=0;j<ClassifierType.values().length;j++){
                classifier=ClassifierType.values()[j];
                try{
                    System.out.println(getTrainTestAcc(classifier,datasetName)+",");
                }catch(Exception e){
                    System.out.println("N/A,");
                }
            }
            System.out.println("++++++");
        }
    }
    //function to print Cv Accuracy data
    public static void printCvAccuracy(String[] dataSets) throws Exception{
        System.out.println("DataSet: ");
        for(int i=0;i<ClassifierType.values().length;i++){
            System.out.println(ClassifierType.values()[i]);
        }
        System.out.println("==========");
        ClassifierType classifier;
        String datasetName;
        for(int i=0;i<dataSets.length;i++){
            datasetName=dataSets[i];
            System.out.println(datasetName.replaceAll("normalised",""));
            for(int j=0;j<ClassifierType.values().length;j++){
                classifier=ClassifierType.values()[j];
                try{
                    System.out.println(getCvAcc(classifier,datasetName)+",");
                }catch(Exception e){
                    System.out.println("N/A,");
                }
            }
            System.out.println("+++++++++");
        }
    }
    //classify method main funciton
    public double classify(String dataSetName,boolean verbose,String outputLocation) throws Exception{
        double acc=0.0;
        ClassifierType[] classifiers=new ClassifierType[this.Classifiers.size()];
        int id=0;
        for(ClassifierType classifier:this.Classifiers){
            classifiers[id++]=classifier;
        }
        double[] actualClassValues=getActualClassValues(dataSetName);
        double[] cvAccuracy=new double[classifiers.length];
        double[][] predictions=new double[classifiers.length][];
        //calculate accuracies
        for(int i=0;i<classifiers.length;i++){
            if(this.ensembleType!=EnsembleType.Equal){
                cvAccuracy[i]=getCvAcc(classifiers[i],dataSetName);
            }
            predictions[i]=getTestPredictions(classifiers[i],dataSetName);
            if(predictions[i].length!=actualClassValues.length){
                throw new Exception("Instance num mismatch between raw data and predictions for "+classifiers[i]+" on "+dataSetName);
            }
        }
        FileWriter out=null;
        StringBuilder st=null;
        if(outputLocation!=null){
            out=new FileWriter(outputLocation);
            st=new StringBuilder();
        }
        //calculate weights
        ArrayList<Integer> bestClassifierIds;
        double[] weights=new double[classifiers.length];
        if(this.ensembleType==EnsembleType.Best){
            double bsfAcc=0.0;
            bestClassifierIds=new ArrayList<>();
            for(int i=0;i<classifiers.length;i++){
                if(cvAccuracy[i]>bsfAcc){
                    bestClassifierIds=new ArrayList<>();
                    bestClassifierIds.add(i);
                    bsfAcc=cvAccuracy[i];
                }else if(cvAccuracy[i]==bsfAcc){
                    bestClassifierIds.add(i);
                }
            }
            for(int i=0;i<bestClassifierIds.size();i++){
                weights[bestClassifierIds.get(i)]=1;
            }
        }else if(this.ensembleType==EnsembleType.Equal){
            for(int i=0;i<classifiers.length;i++){
                weights[i]=1;
            }
        }else if(this.ensembleType==EnsembleType.Prop){
            for(int i=0;i<classifiers.length;i++){
                weights[i]=cvAccuracy[i]/100;
            }
        }else if(this.ensembleType==EnsembleType.TopHalf) {
            int numToUse=classifiers.length/2+1;
            CvCompare[] cvs=new CvCompare[classifiers.length];
            for(int i=0;i<classifiers.length;i++){
                cvs[i]=new CvCompare(cvAccuracy[i],i);
            }
            Arrays.sort(cvs);
            for(int i=0;i<numToUse;i++){
                weights[cvs[cvs.length-1-i].classifierId]=cvs[cvs.length-1-i].cvAccuracy;
            }
        }else{
            throw new Exception("Invalid ensemble type!");
        }
        int correct=0,total=0;
        double classVal,prediction,currentVote,bsfVote;
        ArrayList<Double> bestClassVals;
        HashMap<Double,Double> classAndWeightVotes;
        Random random=new Random();
        //
        for(int i=0;i<actualClassValues.length;i++){
            bsfVote=0.0;
            bestClassVals=new ArrayList<>();
            classVal=actualClassValues[i];
            classAndWeightVotes=new HashMap<>();
            for(int j=0;j<classifiers.length;j++){
                if(weights[j]>0){
                    currentVote=0.0;
                    if(!classAndWeightVotes.containsKey(predictions[j][i])){
                        currentVote=weights[j];
                        classAndWeightVotes.put(predictions[j][i],currentVote);
                    }else{
                        currentVote=classAndWeightVotes.get(predictions[j][i]);
                        currentVote+=weights[j];
                        classAndWeightVotes.put(predictions[j][i],currentVote);
                    }
                    if(currentVote>bsfVote) {
                        bestClassVals = new ArrayList<>();
                        bestClassVals.add(predictions[j][i]);
                        bsfVote = currentVote;
                    }else if(currentVote==bsfVote){
                        bestClassVals.add(predictions[j][i]);
                    }
                }
            }
            if(bestClassVals.size()==1){
                prediction=bestClassVals.get(0);
            }else if(bestClassVals.size()>1){
                prediction = bestClassVals.get(random.nextInt(bestClassVals.size()));
            }else{
                throw new Exception("Error: no classifier was chosen");
            }
            //calculate the final accuracy
            if(prediction==classVal){
                correct++;
            }total++;
            //choose the debug mode
            if(verbose){
                System.out.println((i+1)+","+prediction+","+classVal);
            }
            if(outputLocation!=null){
                st.append(prediction).append(classVal).append("\n");
            }
        }
        //calculate
        acc=(correct+0.0)/total;
        //debug, output the results with specific format
        if(verbose){
            DecimalFormat df=new DecimalFormat("###.##");
            System.out.println("Accuracy: "+correct+"/"+total+","+df.format(acc*100)+"%");
        }
        if(outputLocation!=null){
            out.append(correct+"/"+total+"\n");
            out.append(st);
            out.close();
        }
        return acc;
    }
}
