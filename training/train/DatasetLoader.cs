//https://github.com/virex-84

//DatasetLoader.cs

using System.Text.Json;

namespace TernaryLLM;

public enum DatasetType
{
    JSON,
    CSV
}

/// <summary>
/// Загрузчик датасетов для обучения
/// </summary>
public class DatasetLoader
{
    public List<string> PretrainingData { get; }
    public List<string> ChatTrainingData { get; }

    public DatasetLoader(string pretrainingDataPath, string chatTrainingDataPath, DatasetType typeOfData)
    {
        switch (typeOfData)
        {
            case DatasetType.CSV:
                PretrainingData = GetDataFromCsv(pretrainingDataPath);
                ChatTrainingData = GetDataFromCsv(chatTrainingDataPath);
                break;
            case DatasetType.JSON:
                PretrainingData = GetDataFromJson(pretrainingDataPath);
                ChatTrainingData = GetDataFromJson(chatTrainingDataPath);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(typeOfData), typeOfData, null);
        }
    }

    private static List<string> GetDataFromJson(string path)
    {
        if (!File.Exists(path))
            return new List<string>();

        try
        {
            var json = File.ReadAllText(path);
            if (string.IsNullOrWhiteSpace(json))
                return new List<string>();

            var data = JsonSerializer.Deserialize<List<string>>(json);
            return data ?? new List<string>();
        }
        catch (JsonException)
        {
            return new List<string>();
        }
    }

    private static List<string> GetDataFromCsv(string path)
    {
        var data = new List<string>();
        var lines = File.ReadAllLines(path);
        
        foreach (var line in lines)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                data.Add(line.Trim());
            }
        }
        
        return data;
    }
}
