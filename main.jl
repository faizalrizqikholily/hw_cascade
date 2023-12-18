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

function count_accuracy(kolom, real_class, predicted_class)
    new_data = hcat(kolom, real_class, predicted_class)
    benar = findall(new_data[:, 2] .== new_data[:, 3])
    accuracy = length(benar) / size(new_data, 1) * 100
    accuracy = floor(accuracy)
    return benar, accuracy
end

function update_mu(data_origin, mean_origin, real_class)
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

    benar1, accuracy1 = count_accuracy(data_kolom_1, real_class, predictions_kolom_1)
    benar2, accuracy2 = count_accuracy(data_kolom_2, real_class, predictions_kolom_2)
    benar3, accuracy3 = count_accuracy(data_kolom_3, real_class, predictions_kolom_3)
    benar4, accuracy4 = count_accuracy(data_kolom_4, real_class, predictions_kolom_4)

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

    return selected_benar, accuracies
end

#deserilize data_9m
data_origin = deserialize("data_9m.mat")

#convert data to Float64
data_origin = convert(Array{Float64,2}, data_origin)
mean_origin = calculating_mean(data_origin)

display(mean_origin)

############## Iterasi 1 ##################
selected_benar1, accuracies_iter1 = update_mu(data_origin, mean_origin, data_origin[:, end])
data_iter1 = data_origin[selected_benar1, :]
mean_iter1 = calculating_mean(data_iter1)


println("Akurasi")
display(accuracies_iter1)

println()
println("Update miu Iterasi 1")
display(mean_iter1)

############## Iterasi 2 ##################
selected_benar2, accuracies_iter2 = update_mu(data_iter1, mean_iter1, data_iter1[:, end])
data_iter2 = data_iter1[selected_benar2, :]
mean_iter2 = calculating_mean(data_iter2)

println()
println("Update miu Iterasi 2")
display(mean_iter2)

############## Iterasi 3 ##################
selected_benar3, accuracies_iter3 = update_mu(data_iter2, mean_iter2, data_iter2[:, end])
data_iter3 = data_iter2[selected_benar3, :]
mean_iter3 = calculating_mean(data_iter3)

println()
println("Update miu Iterasi 3")
display(mean_iter3)


############## Iterasi 4 ##################
selected_benar4, accuracies_iter4 = update_mu(data_iter3, mean_iter3, data_iter3[:, end])
data_iter4 = data_iter3[selected_benar4, :]
mean_iter4 = calculating_mean(data_iter4)

println()
println("Update miu Iterasi 4")
display(mean_iter4)