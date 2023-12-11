from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

def main():
    # Load the dataset
    dataset = load_dataset("taaredikahan23/Medical_dataset")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Preprocess function
    def preprocess_function(examples):
        inputs = [doc.split("### Output: ")[0].strip() for doc in examples["text"]]
        outputs = [doc.split("### Output: ")[1].strip() if "### Output: " in doc else "" for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)
        labels = tokenizer(outputs, max_length=128, padding='max_length', truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(dataset["train"], test_size=0.1)

    # Convert lists back to Hugging Face Dataset objects
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    # For testing, use a smaller subset of the dataset
    small_train_dataset = train_dataset.select(range(100))
    small_test_dataset = test_dataset.select(range(100))

    # Preprocess the smaller datasets
    tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True, batch_size=16, num_proc=4)
    tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True, batch_size=16, num_proc=4)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

    # Function for interactive conversation
    def chat_with_model(model, tokenizer):
        model.eval()
        print("You can now chat with the model. Type 'exit' to end the conversation.")
        while True:
            # Get user input
            input_text = input("User: ")
            if input_text.lower() == "exit":
                break
            
            # Tokenize and generate response
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            with torch.no_grad():
                output_ids = model.generate(input_ids, max_length=512)
            
            reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("Model:", reply)

    # Run the interactive chat
    chat_with_model(model, tokenizer)

if __name__ == "__main__":
    main()
