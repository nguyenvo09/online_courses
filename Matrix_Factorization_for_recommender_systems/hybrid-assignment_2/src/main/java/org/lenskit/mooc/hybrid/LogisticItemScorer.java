package org.lenskit.mooc.hybrid;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.LenskitRecommender;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.results.Results;
import org.lenskit.util.collections.LongUtils;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Item scorer that does a logistic blend of a subsidiary item scorer and popularity.  It tries to predict
 * whether a user has rated a particular item.
 */
public class LogisticItemScorer extends AbstractItemScorer {
    private final LogisticModel logisticModel;
    private final BiasModel biasModel;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;

    @Inject
    public LogisticItemScorer(LogisticModel model, UserBiasModel bias, RecommenderList recs, RatingSummary rs) {
        logisticModel = model;
        biasModel = bias;
        recommenders = recs;
        ratingSummary = rs;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        // TODO Implement item scorer
        List<ItemScorer> scorers = recommenders.getItemScorers();
        double intercept = logisticModel.getIntercept();
        RealVector coefficients = logisticModel.getCoefficients();
        List<Result> results = new ArrayList<>();
        double[] d = new double[recommenders.getRecommenderCount()+2];
        for(Long itemId : items){

            d[0] = biasModel.getItemBias(itemId) + biasModel.getUserBias(user) + biasModel.getIntercept();
            d[1] = Math.log10(ratingSummary.getItemRatingCount(itemId));

            for(int i=0; i<recommenders.getRecommenderCount(); i+=1){
                ItemScorer rec = scorers.get(i);

                Result r = rec.score(user, itemId);
                d[i+2] = 0.0;
                if(r != null){
                    d[i+2] = r.getScore() - d[0];
                }
            }
            RealVector prediction_from_other_recs = MatrixUtils.createRealVector(d);
            double vl = intercept + prediction_from_other_recs.dotProduct(coefficients);
//            System.out.println(user+" "+itemId+" "+vl);

            vl = LogisticModel.sigmoid(vl);

            results.add(Results.create(itemId, vl));

        }
        return Results.newResultMap(results);
    }
}
