package org.lenskit.mooc.hybrid;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.LenskitRecommender;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.inject.Transient;
import org.lenskit.util.ProgressLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Trainer that builds logistic models.
 */
public class LogisticModelProvider implements Provider<LogisticModel> {
    private static final Logger logger = LoggerFactory.getLogger(LogisticModelProvider.class);
    private static final double LEARNING_RATE = 0.00005;
    private static final int ITERATION_COUNT = 100;

    private final LogisticTrainingSplit dataSplit;
    private final BiasModel baseline;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;
    private final int parameterCount;
    private final Random random;

    @Inject
    public LogisticModelProvider(@Transient LogisticTrainingSplit split,
                                 @Transient UserBiasModel bias,
                                 @Transient RecommenderList recs,
                                 @Transient RatingSummary rs,
                                 @Transient Random rng) {
        dataSplit = split;
        baseline = bias;
        recommenders = recs;
        ratingSummary = rs;
        parameterCount = 1 + recommenders.getRecommenderCount() + 1;
        random = rng;
    }

    public static void myAssertTrue(boolean condition, String message) throws Exception {
        if (!condition) {
            throw new Exception(message);
        }
        return;

    }
    @Override
    public LogisticModel get(){
        List<ItemScorer> scorers = recommenders.getItemScorers();
        double intercept = 0.0;
        double[] params = new double[parameterCount];
        for(int j=0; j<parameterCount; j+=1){
            params[j] = 0.0;
        }

        List<Rating> train = dataSplit.getTuneRatings();

        LogisticModel current = LogisticModel.create(intercept, params);
        /**Precomputing scores of recommenders with bias */

        ////////////////////////////////////////////////////////////////////////////////
        // START PRE-COMPUTING SCORES OF RECOMMENDERS
        ////////////////////////////////////////////////////////////////////////////////
        int no_recommenders = recommenders.getRecommenderCount();
        int no_ratings = train.size();
        List<List<Double>> pre_computed_scores = new ArrayList<List<Double>>();
        for(int i=0; i<no_recommenders; i+=1){
            List<Double> vls = new ArrayList<>();

            ItemScorer rec_ = scorers.get(i);
            for(int j=0; j<no_ratings; j+=1){
                Rating r = train.get(j);
                Result vl = rec_.score(r.getUserId(), r.getItemId());
                double score_ = 0;
                if(vl!=null){
                    double bias_ = baseline.getIntercept()+
                            baseline.getUserBias(r.getUserId())+
                            baseline.getItemBias(r.getItemId());
                    score_ = vl.getScore()-bias_;
                }

                vls.add(score_);
            }
            pre_computed_scores.add(vls);
        }
        ////////////////////////////////////////////////////////////////////////////////
        // END PRE-COMPUTING
        ////////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////////
        // START GRADIENT ASCENT!!!!
        ////////////////////////////////////////////////////////////////////////////////
        /*Starting gradient ascent!!!!*/
        double partial_intercept = 0.0;
        double[] partial_params = new double[parameterCount];
        List<Double> Losses = new ArrayList<>();
        // TODO Implement model training
        for(int i=0; i<ITERATION_COUNT; i+=1){
            /////////////////////////////////////////
            /// RESET params + data
            ///////////////////////////////////////////

            partial_intercept = 0.0;
            for(int j=0; j<parameterCount; j+=1){
                partial_params[j] = 0.0;
            }
            ////////////////////////////////////////////
            double old_intercept = current.getIntercept();
            RealVector old_betas = current.getCoefficients();
            try {
                myAssertTrue(old_betas.getDimension() == parameterCount, "Mismatch dimension!!!");
            } catch (Exception e) {
                e.printStackTrace();
            }
            //computing gradient!!!
            double LOSS = 0.0;
            for(int k = 0; k<train.size(); k+=1){
                Rating r = train.get(k);
                double y_ui = r.getValue();
                //compute equation inside sigmoid
                double bias = baseline.getIntercept() + baseline.getUserBias(r.getUserId()) + baseline.getItemBias(r.getItemId());
                try {
                    myAssertTrue(Math.abs(y_ui - 1.0) <= 1e-10 || Math.abs(y_ui + 1.0) <= 1e-10,
                            "Somthing wrong with values");
                } catch (Exception e) {
                    e.printStackTrace();
                }
                if(Math.abs(y_ui - (-1.0) ) <= 1e-10){
                    y_ui = 0;
                }

                double[] X = new double[parameterCount];
                X[0] = bias;
                X[1] = Math.log10(ratingSummary.getItemRatingCount(r.getItemId()));

                for(int t = 0; t<no_recommenders; t+=1){
                    double xn = pre_computed_scores.get(t).get(k);
                    X[t+2] = xn;
                }
                try {
                    myAssertTrue(no_recommenders+2 == X.length, "Something wrong here!!!");
                } catch (Exception e) {
                    e.printStackTrace();
                }

                RealVector variables = MatrixUtils.createRealVector(X);
                double inside = y_ui*(old_intercept + variables.dotProduct(old_betas));

                partial_intercept += y_ui - LogisticModel.sigmoid(inside);
                for(int j=0; j<parameterCount; j+=1){
                    partial_params[j] += variables.getEntry(j) * (y_ui - LogisticModel.sigmoid(inside));
                }
                LOSS += y_ui * Math.log(LogisticModel.sigmoid(inside)) + (1-y_ui) * Math.log(1-LogisticModel.sigmoid(inside));
//                System.out.println("Loss of pre of this iteration is: " + LOSS);
            }
            //updating parameters
            intercept = old_intercept + LEARNING_RATE * partial_intercept;
            RealVector partial_params_vec = MatrixUtils.createRealVector(partial_params);
            RealVector new_parms = old_betas.add(partial_params_vec.mapMultiply(LEARNING_RATE));
            params = new_parms.toArray();

            Losses.add(LOSS);
            //Print loss function
//            System.out.println("Loss of pre of this iteration is: " + LOSS);

            current = LogisticModel.create(intercept, params);


        }
//        for(int i=0; i<Losses.size(); i+=1){
//            System.out.println(Losses.get(i));
//        }

        return current;
    }

}
