# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------------------------
# PLOT SAVED PCA COMPONENTS
# -------------------------------------------------------------------

def plot_pca_components(
    pca_output,
    variance_output,
    x_component=1,
    y_component=2
):
    """
    Plot any two of the first five saved PCA components.

    Marker shape represents the true class.
    Color represents the KNN predicted class.
    """

    if x_component < 1 or x_component > 5:
        raise ValueError(
            "x_component must be between 1 and 5."
        )

    if y_component < 1 or y_component > 5:
        raise ValueError(
            "y_component must be between 1 and 5."
        )

    if x_component == y_component:
        raise ValueError(
            "Choose two different PCA components."
        )


    # PCA column names
    x_column = f"PC{x_component}"
    y_column = f"PC{y_component}"


    # Color represents predicted class
    colors = {
        "Control": "#1f77b4",
        "Microgravity": "#d62728"
    }


    # Marker represents true class
    markers = {
        "Control": "o",
        "Microgravity": "s"
    }


    # Get explained variance for selected components
    x_variance = variance_output.loc[
        variance_output["Component"] == x_column,
        "Explained_Variance"
    ].iloc[0]


    y_variance = variance_output.loc[
        variance_output["Component"] == y_column,
        "Explained_Variance"
    ].iloc[0]


    plt.figure(
        figsize=(11, 8)
    )


    # Plot each true / predicted class combination
    for true_label in [
        "Control",
        "Microgravity"
    ]:

        for predicted_label in [
            "Control",
            "Microgravity"
        ]:

            mask = (
                (
                    pca_output["True_Label"]
                    == true_label
                )
                &
                (
                    pca_output["Predicted_Label"]
                    == predicted_label
                )
            )


            if mask.sum() > 0:

                label = (
                    f"True {true_label} | "
                    f"Predicted {predicted_label}"
                )


                plt.scatter(
                    pca_output.loc[
                        mask,
                        x_column
                    ],

                    pca_output.loc[
                        mask,
                        y_column
                    ],

                    c=colors[
                        predicted_label
                    ],

                    marker=markers[
                        true_label
                    ],

                    s=100,

                    edgecolors="black",

                    alpha=0.8,

                    label=label
                )


    # Graph formatting
    plt.title(
        f"{x_column} vs {y_column} of "
        f"Persistence Landscape Vectors"
    )


    plt.xlabel(
        f"{x_column} "
        f"({x_variance:.2f}%)"
    )


    plt.ylabel(
        f"{y_column} "
        f"({y_variance:.2f}%)"
    )


    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )


    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )


    plt.tight_layout()

    plt.show()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":

    # Folder containing saved PCA files
    pca_folder = Path(
        r"C:\Users\g_gar\OneDrive\Documents\GitHub"
        r"\BryanProgramImaging2026\Gabriel"
    )


    # Load saved PCA coordinates
    pca_output = pd.read_csv(
        pca_folder
        / "PCA_First_Five_Components.csv"
    )


    # Load saved explained variance
    variance_output = pd.read_csv(
        pca_folder
        / "PCA_Explained_Variance.csv"
    )


    print(
        "PCA data loaded successfully."
    )


    # Choose PCA components to visualize
    # plot_pca_components(
    #     pca_output,
    #     variance_output,
    #     x_component=1,
    #     y_component=4
    # )
    
    for x_component in range(
    1,
    6
    ):

        for y_component in range(
            x_component + 1,
            6
            ):

            plot_pca_components(
                pca_output,
                variance_output,
                x_component=x_component,
                y_component=y_component
                )
