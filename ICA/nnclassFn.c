/*==================================================================================================
 *  nnclassFn.c
 *
 *  Edited by William Halsey and Scott Rodgers
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *
 *  This file contains
 *		nnclassFn
 *
 *  Lasted Edited: Jul. 9, 2013
 *
 *  Changes made: by William - created this file
 *
 */
void nnclassFn(*testPerf, *rankmat, *rank, train, trainRows, trainCols, test, testRows, testCols, trainClass, answer) {
        // %function [testPerf,rankmat,rank] = nnclassFn(train,test,trainClass,answer)
        // %
        // %Reads in training examples, test examples, class labels of training
        // %examples, and correct class of test examples. Data are in columns of train
        // %and test, and labels are column vectors.
        // % 
        // %Note: You will need to create label vectors. TrainClass is a column 
        // %vector of integers indicating the identity of the training examples. 
        // %e.g. for faces of 3 people with two views each, TrainClass = [1 1 2 2 3 3 ]';
        // %Answer contains the correct labels of the test images, which enables
        // %us to compute percent correct.  
        // %
        // %Gets matrix of normalized dot products. Outputs nearest neighbor
        // %classification of test examples and percent correct.
        // %rankmat gives the top 30 matches for each test image.  rank is a vector
        // %containing the percent of times the correct match is in the top N matches.


        // function [testPerf,rankmat,rank] = nnclassFn(train,test,trainClass,answer)
        // 
        // numTest = size(test,2);
        // numTrain = size(train,2);
    int numTest = testCols;
    int numTrain = trainCols;
        // %Get distances to training examples
        // %dists = eucDist(test,train); %Outputs a Ntest x Ntrain matrix of Euc dist
    transpose(test_t, test);
    transpose(train_t, train);
    dists = -1 * cosFn(output, test_t, train_t, testCols, testRows, trainCols, trainRows);    // dists=-1 * cosFn(test',train');%Outputs a Ntest x Ntrain matrix of cosines

        // %sort the rows of dists to find the nearest training example:
    transpose(dist);
    eigsort(Sdist, dists_t);    // [Sdist,nearest] = sort(dists'); %cols of Sdist are distances in ascend order
                // %1st row of nearest is index of 1st closest training example

                // %Create vector with nearest example, and vector with class label.
        // Nnbr = nearest(1,:);            %First row of nearest contains NN
        // %Nnbr = nearest(2,:);
// testClass = trainClass(Nnbr);

// correct = find( (testClass - answer == 0));
// testPerf = size(correct,1) / size(answer,1)
// if(size(correct,2)>size(correct,1))
//         testPerf = size(correct,2) / size(answer,2)
//         fprintf(1,'check vector orientation')
// end

// %get rank = %correct in top N:
// cumtestPerf=0;
// for i = 1:30
//         rankmat(:,i) = trainClass(nearest(i,:)');
//         correcti = find( (rankmat(:,i) - answer == 0));
//         cumtestPerf = cumtestPerf + size(correcti,1) / size(answer,1);
//         rank(i) = cumtestPerf;
// end
// 
// %For FERET test, want probeID (answer), then rank, then matched ID no.,
// %then FA flag, then "matching score".  This will be a matrix with: 
// %probe  rank    match                           FAflag          matching score
// %i      1       trainClass(nearest(i,:))        Sdist(:,i)>4.7  1./Sdist(:,i)
// %i      2       OR rankmat(i,:)'
// %i      3
// %i      4


    return;
}