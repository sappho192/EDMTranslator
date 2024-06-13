namespace EDMTranslator.Models
{
    /// <summary>
    /// Original model types
    /// Fine-tuned model based on these models are still considered as the original model type.
    /// For example, bert-japanese-base is considered as BERT.
    /// KoGPT2 is still considered as GPT2.
    /// </summary>
    public enum ModelType
    {
        EncoderDecoder,
        BERT,
        GPT2
    }
}
