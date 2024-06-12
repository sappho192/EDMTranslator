namespace EDMTranslator
{
    public interface ITokenizer
    {
        public (int[], int[]) Encode(string sentence);
        public string Decode(uint[] tokens);
    }
}
