# Faizal Rizqi Kholily - 1313621029
# Rasyaad Maulana Khandiyas - 1313621020

using DataFrames, Serialization, Statistics

function calculating_mean(data)
    setosa = data[data[:, end].==1, :]
    versicolor = data[data[:, end].==2, :]
    virginica = data[data[:, end].==3, :]

    mean_setosa = mean(setosa[:, 1:4], dims=1)
    mean_versicolor = mean(versicolor[:, 1:4], dims=1)
    mean_virginica = mean(virginica[:, 1:4], dims=1)

    concatenated_means = [
        mean_setosa
        mean_versicolor
        mean_virginica
    ]

    return concatenated_means
end

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
    new_data = hcat(kolom, data_origin[:, end], predicted_class)
    benar = findall(new_data[:, 2] .== new_data[:, 3])  # Extract indices where benar is true
    accuracy = length(benar) / size(new_data, 1) * 100
    return benar, accuracy
end

#deserilize data_9m
data_origin = deserialize("data_9m.mat")

#convert data to Float64
data_origin = convert(Array{Float64,2}, data_origin)

mean_origin = calculating_mean(data_origin)

# Pisahkan data utuh menjadi 4 databaru berdasarkan kolom
data_kolom_1 = data_origin[:, 1]
data_kolom_2 = data_origin[:, 2]
data_kolom_3 = data_origin[:, 3]
data_kolom_4 = data_origin[:, 4]

# Pisahkan data hasil concatenated_means menjadi 4 data baru berdasarkan kolom
mean_kolom_1 = mean_origin[:, 1]
mean_kolom_2 = mean_origin[:, 2]
mean_kolom_3 = mean_origin[:, 3]
mean_kolom_4 = mean_origin[:, 4]

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
selected_rows_iter1 = data_origin[selected_benar, :]

selected_rows_iter1 = calculating_mean(selected_rows_iter1)

display(mean_origin)
display(selected_rows_iter1)