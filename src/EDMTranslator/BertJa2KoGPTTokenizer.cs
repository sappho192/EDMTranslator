using BertJapaneseTokenizer;
using Tokenizers.DotNet;

namespace EDMTranslator;

public class BertJa2KoGPTTokenizer
{
    private readonly BertJapaneseTokenizer.BertJapaneseTokenizer sourceTokenizer;
    private readonly Tokenizers.DotNet.Tokenizer targetTokenizer;

    // Unused & Should not be used
    private BertJa2KoGPTTokenizer()
    {
        sourceTokenizer = new BertJapaneseTokenizer.BertJapaneseTokenizer("", "");
        targetTokenizer = new Tokenizers.DotNet.Tokenizer();
    }

    public BertJa2KoGPTTokenizer(string dictDir, string vocabPath)
    {
        sourceTokenizer = new BertJapaneseTokenizer.BertJapaneseTokenizer(dictDir, vocabPath);
        targetTokenizer = new Tokenizers.DotNet.Tokenizer();
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
