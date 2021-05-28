import matplotlib.pyplot as plt
import seaborn as sns


# This modules consists of functions for visualize data


# TODO: box plot, count the total number of any target for the presentation, detect if the dataset is balanced
def count_plot_target_class(data_frame, target_column, generated_image_name):
    """Generate an image regarding the distribution of given column of a dataframe"""
    print(data_frame.groupby([target_column]).size())  # print the sum of every class

    sns.countplot(data=data_frame, x=data_frame[target_column])
    plt.title('Display the distribution of ' + target_column + ' class')
    plt.xlabel('Target Name: ' + target_column)
    plt.ylabel('Count')
    save_plot_as_image(generated_image_name)
    plt.show()


def save_plot_as_image(image_name):
    plt.savefig('presentation/images/' + image_name + '.png', bbox_inches='tight')

# TODO: Dataset desc for the presentation

# TODO: plot the Bag Of Words

# TODO: count the words
