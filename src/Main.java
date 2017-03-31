public class Main {
    public static String[] DATASET_LIST_SDM = {
            // <editor-fold defaultstate="collapsed" desc="Sorted approximately in ascending order of run-time when generating original results">
            "ItalyPowerDemand",
            "SonyAIBORobotSurface",
            "TwoLeadECG",
            "SonyAIBORobotSurfaceII",
            "MoteStrain",
            "ECGFiveDays",
            "CBF",
            "GunPoint",
            "DiatomSizeReduction",
            "SyntheticControl",
            "Plane",
            "FaceFour",
            "MiddlePhalanxOutlineAgeGroup",
            "MiddlePhalanxTW",
            "DistalPhalanxTW",
            "ProximalPhalanxOutlineAgeGroup",
            "DistalPhalanxOutlineAgeGroup",
            "normalisedCoffee",                 // as discussed in the paper, Coffee, etc were normalised
            "ProximalPhalanxTW",
            "ToeSegmentation1",
            "Symbols",
            "ToeSegmentation2",
            "MedicalImages",
            "FacesUCR",
            "BirdChicken",
            "ArrowHead",
            "DistalPhalanxOutlineCorrect",
            "ProximalPhalanxOutlineCorrect",
            "MiddlePhalanxOutlineCorrect",
            "Trace",
            "BeetleFly",
            "Lightning7",
            "SwedishLeaf",
            "normalisedOliveOil",               // normalised, as discussed
            "normalisedBeef",                   // normalised, as discussed
            "FaceAll",
            "Herring",
            "WordSynonyms",
            "TwoPatterns",
            "ChlorineConcentration",
            "wafer",
            "Car",
            "Lightning2",
            "fiftywords",
            "fish",
            "OSULeaf",
            "Cricket_Z",
            "Cricket_Y",
            "yoga",
            "MALLAT",
            "Cricket_X",
            "UWaveGestureLibrary_Z",
            "UWaveGestureLibrary_Y",
            "UWaveGestureLibrary_X",
            "ElectricDevices",
            "Earthquakes",
            "CinC_ECG_torso",
            "Haptics",
            "Computers",
            "LargeKitchenAppliances",
            "RefrigerationDevices",
            "ScreenType",
            "SmallKitchenAppliances",
            "Adiac",
            "InlineSkate",
            "StarLightCurves",
            "NonInvasiveFatalECG_Thorax2",
            "NonInvasiveFatalECG_Thorax1",
            "FordA",
            "FordB",
            "normalisedWorms",              // new datasets were normalised too
            "normalisedWormsTwoClass",      // normalised, as discussed
            
            // </editor-fold>
    };
    
    public static void main(String[] args) {
        //initial the classifiers
        FlatCOTE flat=new FlatCOTE();
        flat.chooseAll();
        //elastic ensemble classifier --- combination of some NN algorithms
        FlatCOTE ee=new FlatCOTE();
        ee.chooseNN();
        FlatCOTE shapelet=new FlatCOTE();
        shapelet.chooseShapelet();
        FlatCOTE ps=new FlatCOTE();
        ps.choosePS();
        FlatCOTE acf=new FlatCOTE();
        acf.chooseACF();
        String[] datasets = {"normalisedWorms","normalisedWormsTwoClass"};
        System.out.println("Datasets,Flat-COTE,Elastic,Shapelet,PS,ACF");
        for(int d=0;d<datasets.length;d++){
            String dataset=datasets[d];
            System.out.println(dataset);
            try{
                //System.out.println(1);
                double acc_flat=flat.classify(dataset,false,null);
                //System.out.println(2);
                double acc_ee=ee.classify(dataset,false,null);
                //System.out.println(3);
                double acc_shapelet=shapelet.classify(dataset,false,null);
                //System.out.println(4);
                double acc_ps=ps.classify(dataset,false,null);
                //System.out.println(5);
                double acc_acf=acf.classify(dataset,false,null);
                String output=dataset.replaceAll("normalised","")+":\n"+"Flat:"+acc_flat+"\nEE:"+acc_ee+"\nShapelet:"+acc_shapelet+"\nPS:"+acc_ps+"\nACF:"+acc_acf+"\n";
                System.out.println(output);
            }catch (Exception e) {
                System.out.println(dataset+" missing");
            }
        }
    }
}
