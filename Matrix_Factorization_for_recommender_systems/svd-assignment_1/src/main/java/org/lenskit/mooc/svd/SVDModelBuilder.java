package org.lenskit.mooc.svd;

import org.apache.commons.math3.linear.*;
import org.lenskit.bias.BiasModel;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonTypes;
import org.lenskit.data.ratings.Rating;
import org.lenskit.inject.Transient;
import org.lenskit.util.io.ObjectStream;
import org.lenskit.util.keys.FrozenHashKeyIndex;
import org.lenskit.util.keys.KeyIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.Iterator;

/**
 * Model builder that computes the SVD model.
 */
public class SVDModelBuilder implements Provider<SVDModel> {
    private static final Logger logger = LoggerFactory.getLogger(SVDModelBuilder.class);

    private final DataAccessObject dao;
    private final BiasModel baseline;
    private final int featureCount;

    /**
     * Construct the model builder.
     * @param dao The data access object.
     * @param bias The bias model to use as a baseline.
     * @param nfeatures The number of latent features to train.
     */
    @Inject
    public SVDModelBuilder(@Transient DataAccessObject dao,
                           @Transient BiasModel bias,
                           @LatentFeatureCount int nfeatures) {
        this.dao = dao;
        baseline = bias;
        featureCount = nfeatures;
    }

    /**
     * Build the SVD model.
     *
     * @return A singular value decomposition recommender model.
     */
    @Override
    public SVDModel get() {
        // Create index mappings of user and item IDs.
        // You can use these to find row and columns in the matrix based on user/item IDs.
        KeyIndex userIndex = FrozenHashKeyIndex.create(dao.getEntityIds(CommonTypes.USER));
        KeyIndex itemIndex = FrozenHashKeyIndex.create(dao.getEntityIds(CommonTypes.ITEM));

        // We have to do 2 things:
        // First, prepare a matrix containing the rating data.
        // You will implement createRatingMatrix
        RealMatrix matrix = createRatingMatrix(userIndex, itemIndex);

        long startTime = System.currentTimeMillis();


        // Second, compute its factorization
        logger.info("factorizing matrix");
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
        logger.info("decomposed matrix has rank {}", svd.getRank());

        long estimatedTime = System.currentTimeMillis() - startTime;
//        System.out.println("Running time of SVD decomposition is: " + estimatedTime/(1000.0) + " seconds");

        // Third, truncate the decomposed matrix
        RealMatrix userMatrix = svd.getU();
        RealMatrix itemMatrix = svd.getV();
        RealVector weights = new ArrayRealVector(svd.getSingularValues());

        if (featureCount > 0) {
            logger.info("truncating matrix to {} features", featureCount);
            // TODO Use the getSubMatrix method to truncate the user and item matrices
            int no_user = userIndex.size();
            int no_items = itemIndex.size();
//            System.out.println("So luong user: " + no_user + " so luong items: " + no_items);
            userMatrix = userMatrix.getSubMatrix(0, no_user-1, 0, featureCount-1);
            int no_rows = itemMatrix.getRowDimension();
            int no_colums = itemMatrix.getColumnDimension();
            int no_rank = svd.getRank();
            itemMatrix = itemMatrix.getSubMatrix(0, no_items-1, 0, featureCount-1);

            weights = weights.getSubVector(0, featureCount);
        }

        return new SVDModel(userIndex, itemIndex,
                            userMatrix, itemMatrix,
                            weights);
    }

    /**
     * Build a rating residual matrix from the rating data.  Each user's ratings are
     * normalized by subtracting a baseline score (usually a mean).
     *
     * @param userIndex The index mapping of user IDs to row numbers.
     * @param itemIndex The index mapping of item IDs to column numbers.
     * @return A matrix storing the <i>normalized</i> user ratings.
     */
    private RealMatrix createRatingMatrix(KeyIndex userIndex, KeyIndex itemIndex) {
        final int nusers = userIndex.size();
        final int nitems = itemIndex.size();

        // Create a matrix with users on rows and items on columns
        logger.info("creating {} by {} rating matrix", nusers, nitems);
        RealMatrix matrix = MatrixUtils.createRealMatrix(nusers, nitems);

        // populate it with data
        try (ObjectStream<Rating> ratings = dao.query(Rating.class).stream()) {
            // TODO Put this user's ratings into the matrix
            Iterator<Rating> it = ratings.iterator();
            while(it.hasNext()){
                Rating curr = it.next();
                long itemId = curr.getItemId();
                long userId = curr.getUserId();

                int row = userIndex.getIndex(userId);
                int col = itemIndex.getIndex(itemId);
                double b = baseline.getIntercept()+baseline.getItemBias(itemId)+baseline.getUserBias(userId);
                matrix.setEntry(row, col, curr.getValue()-b);
            }
        }
        //normalizing stuff!!!
        //it looks this framework will do normalizing for us!!! haizzz
//        if(1 == 1){
//            int x = 0;
//            x+=1;
//        }
//        System.out.println("Done loading rating matrix!!!!");
        return matrix;
    }
}
