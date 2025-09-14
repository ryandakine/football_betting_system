# Cursor AI Rules for MLB Betting System

## Project Context
This is a sophisticated MLB sports betting system that combines:
- Mathematical edge detection from odds analysis
- YouTube sentiment analysis for contrarian signals
- Kelly criterion optimization
- N8N workflow automation
- Supabase data storage and reporting

## Code Style & Standards
- Use TypeScript-style JSDoc comments for all functions
- Implement comprehensive error handling for betting data
- Add detailed logging for debugging in N8N environment
- Follow functional programming patterns where appropriate
- Use descriptive variable names (e.g., `valueOpportunities` not `vo`)

## Betting System Conventions
- Sentiment scores: -1 (very bearish) to +1 (very bullish)
- Odds format: American odds (e.g., -110, +150)
- Team names: Handle variations (Yankees, NY Yankees, New York Yankees)
- Date format: YYYY-MM-DD for consistency
- Currency: USD for all monetary calculations

## Error Handling Patterns
```javascript
// Always validate input data
if (!items || !Array.isArray(items) || items.length === 0) {
  console.log('❌ No items to process');
  return [{ json: { error: 'No data' } }];
}

// Wrap processing in try-catch
try {
  // Processing logic
} catch (error) {
  console.log('❌ Error:', error.message);
  return [{ json: { error: error.message } }];
}
```

## Logging Standards
- Use emojis for visual scanning: 📊 (data), ✅ (success), ❌ (error), 🎯 (target)
- Include timestamps and data counts
- Log both input and output for debugging

## N8N Integration Patterns
- Input: `$input.all()` for array processing
- Output: `[{ json: processedData }]` format
- Handle missing fields gracefully
- Validate data types before processing

## Supabase Integration
- Use consistent column naming
- Format dates as YYYY-MM-DD
- Include metadata fields (timestamp, source, confidence)
- Handle empty/null values appropriately

## Performance Considerations
- Process data in batches for large datasets
- Cache frequently accessed data
- Use efficient algorithms for sentiment analysis
- Minimize API calls where possible

## Testing Approach
- Create mock data for all scenarios
- Test edge cases (missing data, malformed odds)
- Validate sentiment score ranges
- Test Kelly criterion calculations
- Verify Supabase output format

## Documentation Requirements
- Document all function parameters and return values
- Include usage examples for complex functions
- Explain betting logic and mathematical concepts
- Provide troubleshooting guides for common issues
