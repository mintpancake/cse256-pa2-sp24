import matplotlib.pyplot as plt
import torch


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @torch.no_grad()
    def sanity_check(self, sentence, block_size, savename=""):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        self.model.eval()
        _, _, attn_maps = self.model(input_tensor)
        self.model.train()
        attn_maps = attn_maps.view(
            -1, attn_maps.shape[2], attn_maps.shape[3], attn_maps.shape[4]
        )

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = (
                attn_map.squeeze(0).detach().cpu().numpy()
            )  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(
                total_prob_over_rows > 1.01
            ):
                print(
                    "Failed normalization test: probabilities do not sum to 1.0 over rows"
                )
                print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap="hot", interpolation="nearest")
            ax.xaxis.tick_top()
            fig.colorbar(cax, ax=ax)
            plt.title(f"Attention Map {j + 1}")

            # Save the plot
            print(f"attn_maps/{savename}_{self.model.__class__.__name__}_{j + 1}.png")
            plt.savefig(
                f"attn_maps/{savename}_{self.model.__class__.__name__}_{j + 1}.png"
            )
