import os
import json
import matplotlib.pyplot as plt

def list_performance(directory):
    records = []

    for filename in os.listdir(directory):
        if(filename.endswith(".json")):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                records.append(data)

    return records

def contain_substrings(string, substrings):
    for substring in substrings:
        if substring in string:
            return True
    
    return False

def show_result(records, substrings):
    sorted_records = sorted(records, key=lambda x: x["name"])
    for record in sorted_records:
        name = record["name"]
        if not contain_substrings(name, substrings):
            continue
        training_epoch = [epoch["training_epoch"] for epoch in record["training_records"]]
        average_accuracies = [sum(epoch["accuracy"].values()) / len(epoch["accuracy"]) for epoch in record["training_records"]]

        print(name+": "+str(average_accuracies[2]))
        plt.plot(training_epoch, average_accuracies, label=name)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    # plt.title("Epoch-Accuracy Curve for "+substrings[0])
    plt.title("Epoch-Accuracy Curve")

    # plt.legend(loc="upper right", bbox_to_anchor=(1,0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
    # plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    directory = "./records/records_CIFAR10/"
    directory = "./records/records_CIFAR10_10iters/"

    records = list_performance(directory)
    features_strings = ["layers", "original"]
    show_result(records, features_strings)

    features_strings = ["c1_12", "c3_28", "13_layers","original"]
    show_result(records, features_strings)

