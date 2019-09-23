/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package perceptron_ocr;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Black
 */
public class Perceptron_OCR {

    /**
     * @param args the command line arguments
     */
    static LinkedList<String> images = new LinkedList<>();
    static LinkedList<String> test_images = new LinkedList<>();
    static LinkedList<Double> t_weights = new LinkedList<>();
    static LinkedHashMap<Integer,LinkedList<Double>> all_weights = new LinkedHashMap<>();
    static LinkedHashMap<Integer,LinkedList<Double>> pixels = new LinkedHashMap<>();
    static double Learning_rate = .01,def_weight = .01;
    static int accuracy = 0;
    static int no_of_weights;
    
    
    public static void main(String[] args) {
        try 
        {
            //optic_trainCopy.txt optdigits.tra optic_train.txt
            File f = new File("src/data/optdigits.tra");
            Path p = f.toPath();
            images = new LinkedList<>(Files.readAllLines(p));
            

            f = new File("src/data/optdigits.tes");
            p = f.toPath();            
            test_images = new LinkedList<>(Files.readAllLines(p));


            //populate pixels
            for(int i = 0; i < images.size();i++)
            {
                String[] splitted_s = images.get(i).split(",");
                pixels.put(i, new LinkedList<>());
                for(int x = 0; x < splitted_s.length;x++)
                {
                    pixels.get(i).add(Double.parseDouble(splitted_s[x]));
                }
            }
            //no of weights = no of inputs
            no_of_weights = images.get(0).split(",").length - 1;
            single_perceptron();
            
        } catch (Exception ex) {
            Logger.getLogger(Perceptron_OCR.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    private static void single_perceptron()
    {
        //populate weight
            
            
            for(int i = 0; i <= 9; i++)
            {
                //populate the perceptrons 
                all_weights.put(i, new LinkedList<>());
                for(int x = 0; x < no_of_weights;x++)
                {
                    //populate def weight for each perceptron based on number image pixels
                    //
                    all_weights.get(i).add(def_weight);
                }
            }
            
            //number of epochs 40
            for(int x = 0; x < 40;x++)
            {
                //no of single perceptrons 9
                for(int i = 0;i < 10;i++)
                {
                    //train each perceptron to handle a single number
                    for(Map.Entry<Integer,LinkedList<Double>> entry : pixels.entrySet())
                    {
                        single_train_weight(entry.getValue(), i,false,x,Learning_rate);
                        
                    }
                    Learning_rate *= .5; 
                }
            }
            
            System.out.println("Weights");
            for(Map.Entry<Integer, LinkedList<Double>> entry : all_weights.entrySet())
            {
                System.out.println(entry);
            }

            single_test(test_images,all_weights);
    }
    
    public static void single_train_weight(LinkedList<Double> pixels,int percep_no,boolean debug,int nb_epoch,double learning_rate)
    {
        
        double target = pixels.get(pixels.size() - 1);
        boolean isRight = percep_no == target;
        
            double output = .0;
            for(int i = 0; i < (pixels.size() - 1); i++)
            {
                output += (pixels.get(i)*all_weights.get(percep_no).get(i));
            }

            double transfer = sigmoid_transfer(output);
            
            if(debug)
            {
                System.out.println("Working on Perceptron No "+percep_no);
                System.out.println("Target : "+target+" Sigmoid : "+sigmoid_transfer(output)+" on Perceptron "+percep_no);
            }
            
            Double error_rate = error_rate(transfer,isRight);
            if(debug)
            {
                System.out.println("Error Rate "+error_rate.toString());
            }
            //update weights 
            for(int i = 0; i < (pixels.size() - 1);i++)
            {
                double old_weight = all_weights.get(percep_no).get(i);
                double updated_w = update_weight(all_weights.get(percep_no).get(i), learning_rate, error_rate, pixels.get(i));
                all_weights.get(percep_no).add(i, updated_w);
                all_weights.get(percep_no).remove(i+1);
                if(debug)
                {
                    System.out.println("Weight "+old_weight);
                    System.out.println("Updated Weight "+updated_w);
                }
            }
            if(debug) System.out.println(">Epoch "+ nb_epoch +" ,learning rate "+learning_rate+",error "+error_rate);   
    }
    
    private static void single_test(LinkedList<String> test_pixels,LinkedHashMap<Integer,LinkedList<Double>> weights)
    {
        int count = 0;
        for(String s : test_pixels)
        {
            System.out.println(s);
            String[] split = s.split(",");
            double old_val = Double.MIN_VALUE;
            String prediction = "";
            int pred_no = -1;
            for(Map.Entry<Integer,LinkedList<Double>> entry : weights.entrySet())
            {
                double output = .0;
                for(int i = 0; i < (split.length - 1) ;i++)
                {
                    output += (Double.parseDouble(split[i]) * entry.getValue().get(i));
                }
                output = sigmoid_transfer(output);
                //update predictions as confidence level changes
                if(output > old_val)
                {
                    old_val = output;
                    prediction = "Prediction "+entry.getKey();
                    pred_no = entry.getKey();
                }
            }
            
            if(Integer.parseInt(split[split.length - 1]) == pred_no)
            {
                count+=1;
            }
            System.out.println(prediction);
            System.out.println("=======================================================");
        }
        System.out.println("Correct "+count);
        Double perc = ((double)count/test_pixels.size())*100.0;
        System.out.println("% "+perc);
    }
    
    private static double sigmoid_transfer(double activation)
    {
        
	return (1.0 / (1.0 + Math.exp(-activation)));
    }
    
    private static double error_rate(double output,boolean isRight)
    {    
        return isRight? (1 - output): (0 - output);
    }
    
    private static double update_weight(double old_weight,double learning_rate,double error,double input)
    {
         return old_weight + (learning_rate * error * input);
    }
}
