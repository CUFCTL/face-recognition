% Implementation of a face recognition system using PCA, LDA, and ICA.
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function [] = face_rec(path_train, path_test, pca, lda, ica, verbose)

addpath PCA LDA ICA

% initialize algorithms
algorithms = [ ...
    struct( ...
        'name', 'PCA', ...
        'enabled', pca ...
    ), ...
    struct( ...
        'name', 'LDA', ...
        'enabled', lda ...
    ), ...
    struct( ...
        'name', 'ICA', ...
        'enabled', ica ...
    ) ...
];

% create training set, test set
[TrainFiles, X, Class_number] = CreateDatabase(path_train);
[TestFiles, X_test] = CreateDatabase(path_test);

% subtract mean from training set, test set
m = mean(X, 2);

X = X - m * ones(1, size(X, 2));
X_test = X_test - m * ones(1, size(X_test, 2));

% run each algorithm
for i = 1 : length(algorithms)
    algo = algorithms(i);

    if algo.enabled
        % compute projection matrix
        if strcmp(algo.name, 'PCA')
            W = EigenfaceCore(X);
        elseif strcmp(algo.name, 'LDA')
            W = FisherfaceCore(X, Class_number);
        elseif strcmp(algo.name, 'ICA')
            W = fastica(X')';
        end

        % compute projected images
        P = W' * X;

        % compute projected test images
        P_test = W' * X_test;

        % perform recognition on each image in the test set
        rec_labels = cell(size(TestFiles));

        for j = 1 : size(TestFiles, 1)
            rec_index = Recognition(P, P_test(:, j));

            rec_labels{j} = TrainFiles(rec_index).label;
        end

        % compute accuracy
        num_correct = 0;

        for j = 1 : size(TestFiles, 1)
            if strcmp(rec_labels{j}, TestFiles(j).label)
                num_correct = num_correct + 1;
            end
        end

        accuracy = num_correct / size(TestFiles, 1) * 100;

        % print results
        if verbose
            fprintf('%s\n', algo.name)

            for j = 1 : size(TestFiles, 1)
                strerror = '';
                if ~strcmp(rec_labels{j}, TestFiles(j).label)
                    strerror = ' (!)';
                end

                fprintf('  %-10s -> %-4s %s\n', TestFiles(j).name, rec_labels{j}, strerror);
            end

            fprintf('  %d / %d matched, %.2f%%\n', num_correct, size(TestFiles, 1), accuracy);
            fprintf('\n');
        else
            fprintf('%.2f\n', accuracy);
        end
    end
end
