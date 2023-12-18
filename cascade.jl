# Faizal Rizqi Kholily - 1313621029
# Rasyaad Maulana Khandiyas - 1313621020

using DataFrames, Serialization, Statistics


function euclidean_distance(v1, v2)
    return sum(abs.(v1 - v2))
end

function predict_class(data_utuh, concatenated_means)
    num_rows = size(data_utuh, 1)
    num_classes = size(concatenated_means, 1)

    predictions = []

    for i in 1:num_rows
        distances = []
        for j in 1:num_classes
            distance = euclidean_distance(data_utuh[i, :], concatenated_means[j, :])
            push!(distances, distance)
        end
        closest_class = argmin(distances)
        push!(predictions, closest_class)
    end

    return predictions
end

function count_accuracy(kolom, predicted_class)
    new_data = hcat(kolom, data[:, end], predicted_class)
    benar = findall(new_data[:, 2] .== new_data[:, 3])  # Extract indices where benar is true
    accuracy = length(benar) / size(new_data, 1) * 100
    return benar, accuracy
end

#deserilize data_9m
data = deserialize("data_9m.mat")

#convert data to Float64
data = convert(Array{Float64,2}, data)

# Filter rows where last column = 1
data_1 = data[data[:, end].==1, :]
data_2 = data[data[:, end].==2, :]
data_3 = data[data[:, end].==3, :]

# Compute mean of column 1, 2, 3, 4 from data_1
mean_1 = mean(data_1[:, 1:4], dims=1)
mean_2 = mean(data_2[:, 1:4], dims=1)
mean_3 = mean(data_3[:, 1:4], dims=1)

concatenated_means = [
    mean_1
    mean_2
    mean_3
]

# Pisahkan data utuh menjadi 4 databaru berdasarkan kolom
data_kolom_1 = data[:, 1]
data_kolom_2 = data[:, 2]
data_kolom_3 = data[:, 3]
data_kolom_4 = data[:, 4]

# Pisahkan data hasil concatenated_means menjadi 4 data baru berdasarkan kolom
mean_kolom_1 = concatenated_means[:, 1]
mean_kolom_2 = concatenated_means[:, 2]
mean_kolom_3 = concatenated_means[:, 3]
mean_kolom_4 = concatenated_means[:, 4]

# Hitung prediksi kelas untuk setiap kolom
predictions_kolom_1 = predict_class(data_kolom_1, mean_kolom_1)
predictions_kolom_2 = predict_class(data_kolom_2, mean_kolom_2)
predictions_kolom_3 = predict_class(data_kolom_3, mean_kolom_3)
predictions_kolom_4 = predict_class(data_kolom_4, mean_kolom_4)

benar1, accuracy1 = count_accuracy(data_kolom_1, predictions_kolom_1)
benar2, accuracy2 = count_accuracy(data_kolom_2, predictions_kolom_2)
benar3, accuracy3 = count_accuracy(data_kolom_3, predictions_kolom_3)
benar4, accuracy4 = count_accuracy(data_kolom_4, predictions_kolom_4)

display(hcat(accuracy1, accuracy2, accuracy3, accuracy4))
display(size(benar3))

accuracies = [accuracy1, accuracy2, accuracy3, accuracy4]
sorted_accuracies = sort(accuracies, rev=true)

if sorted_accuracies[1] == accuracy1
    selected_benar = benar1
elseif sorted_accuracies[1] == accuracy2
    selected_benar = benar2
elseif sorted_accuracies[1] == accuracy3
    selected_benar = benar3
elseif sorted_accuracies[1] == accuracy4
    selected_benar = benar4
end

# selecting all rows based on selected_benar
selected_rows_iter1 = data[selected_benar, :]

setosa_iter1 = selected_rows_iter1[selected_rows_iter1[:, end].==1, :]
versicolor_iter1 = selected_rows_iter1[selected_rows_iter1[:, end].==2, :]
virginica_iter1 = selected_rows_iter1[selected_rows_iter1[:, end].==3, :]

mean_setosa_iter1 = mean(setosa_iter1[:, 1:4], dims=1)
mean_versicolor_iter1 = mean(versicolor_iter1[:, 1:4], dims=1)
mean_virginica_iter1 = mean(virginica_iter1[:, 1:4], dims=1)

concatenated_means_iter1 = [
    mean_setosa_iter1
    mean_versicolor_iter1
    mean_virginica_iter1
]

display(concatenated_means)
display(concatenated_means_iter1)