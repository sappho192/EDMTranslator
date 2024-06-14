using BertJapaneseTokenizer;
using Tokenizers.DotNet;

namespace EDMTranslator.Tokenization;

public class BertJa2KoGPTTokenizer : ITokenizer
{
    private readonly BertJapaneseTokenizer.BertJapaneseTokenizer sourceTokenizer;
    private readonly Tokenizer targetTokenizer;

    // Unused & Should not be used
    private BertJa2KoGPTTokenizer()
    {
        sourceTokenizer = new BertJapaneseTokenizer.BertJapaneseTokenizer("", "");
        targetTokenizer = new Tokenizer("");
    }

    public BertJa2KoGPTTokenizer(string encoderDictDir, string encoderVocabPath,
        string decoderVocabPath)
    {
        sourceTokenizer = new BertJapaneseTokenizer.BertJapaneseTokenizer(
            dictPath: encoderDictDir, vocabPath: encoderVocabPath);
        targetTokenizer = new Tokenizer(vocabPath: decoderVocabPath);
    }

    public (int[], int[]) Encode(string sentence)
    {
        (var inputIds, var attentionMask) = sourceTokenizer.EncodePlus(sentence);
        return (inputIds, attentionMask);
    }

    public string Decode(uint[] tokens)
    {
        return targetTokenizer.Decode(tokens);
    }
}
