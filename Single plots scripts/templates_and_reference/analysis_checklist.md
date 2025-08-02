# Analysis Development Checklist

## Overview

This checklist provides a step-by-step guide for creating new glacier albedo analysis scripts using the established template and structure. Follow this checklist to ensure consistency, quality, and completeness.

## Pre-Development Setup

### ✅ **1. Environment Preparation**
- [ ] Verify Python environment has all required packages
- [ ] Confirm access to glacier data files (MODIS and AWS)
- [ ] Check that `output_manager.py` is available in the working directory
- [ ] Review existing analysis files for similar functionality

### ✅ **2. Analysis Planning**
- [ ] Define the specific research question or objective
- [ ] Determine the type of visualization needed (scatterplot, bar chart, time series, etc.)
- [ ] Identify which glaciers will be included (athabasca, haig, coropuna)
- [ ] Choose appropriate MODIS methods for the analysis
- [ ] Plan the statistical analysis approach

### ✅ **3. Reference Materials**
- [ ] Read `CODE_STRUCTURE_REFERENCE.md` for implementation guidelines
- [ ] Review `config_examples.py` for configuration patterns
- [ ] Examine existing analysis files with similar objectives
- [ ] Identify the most appropriate template configuration

## Development Phase

### ✅ **4. File Creation and Basic Setup**
- [ ] Copy `analysis_template.py` to new filename with descriptive name
- [ ] Update the file header docstring with analysis description
- [ ] List key features and objectives in the docstring
- [ ] Update author and date fields

### ✅ **5. Configuration Customization**
- [ ] Update `CONFIG['output']['analysis_name']` with lowercase identifier
- [ ] Set appropriate `plot_filename` with descriptive name
- [ ] Write detailed `analysis_type` and `description` in summary template
- [ ] Choose appropriate figure size from `config_examples.py`
- [ ] Select relevant color scheme for the analysis type
- [ ] Configure quality filters based on data requirements

### ✅ **6. Data Processing Implementation**
- [ ] Review and customize `DataLoader` class if needed
- [ ] Implement pixel selection strategy in `PixelSelector`
- [ ] Customize `DataProcessor.merge_and_process()` for analysis needs
- [ ] Implement analysis-specific statistical calculations
- [ ] Add any additional data preprocessing steps

### ✅ **7. Visualization Implementation**
- [ ] Rename `AnalysisVisualizer` class to match analysis type
- [ ] Implement `create_visualization()` method with appropriate subplot layout
- [ ] Create individual subplot methods (`_create_subplot()`)
- [ ] Add proper axis labels, titles, and legends
- [ ] Implement color scheme and styling
- [ ] Add statistical annotations (correlation values, sample sizes, etc.)

### ✅ **8. Summary and Documentation**
- [ ] Customize `generate_summary_and_readme()` function
- [ ] Define analysis-specific statistics to collect
- [ ] Implement key findings generation logic
- [ ] Add relevant additional information for README
- [ ] Ensure all important metrics are captured in summary

## Testing and Validation

### ✅ **9. Functionality Testing**
- [ ] Test data loading for all three glaciers individually
- [ ] Verify pixel selection logic works correctly
- [ ] Confirm data merging produces expected results
- [ ] Check statistical calculations for accuracy
- [ ] Validate outlier filtering is working properly

### ✅ **10. Visualization Testing**
- [ ] Run analysis and verify plots are generated correctly
- [ ] Check that all subplots display appropriate data
- [ ] Confirm color scheme is applied consistently
- [ ] Verify axis labels and titles are descriptive
- [ ] Ensure legends and annotations are readable

### ✅ **11. Output Testing**
- [ ] Verify output directory structure is created correctly
- [ ] Check that plots are saved to `plots/` subdirectory
- [ ] Confirm `summary.txt` contains relevant statistics
- [ ] Review `README.md` for completeness and accuracy
- [ ] Validate file naming conventions are followed

### ✅ **12. Data Validation**
- [ ] Check that albedo values are in expected range (0-1)
- [ ] Verify date ranges are reasonable for each glacier
- [ ] Confirm sample sizes are adequate for statistical analysis
- [ ] Validate that correlation values and error metrics are realistic
- [ ] Check for any obvious data quality issues

## Code Quality Assurance

### ✅ **13. Code Review**
- [ ] Verify all functions have proper docstrings
- [ ] Check that type hints are included for all parameters
- [ ] Confirm error handling is implemented throughout
- [ ] Ensure logging statements provide useful information
- [ ] Review variable names for clarity and consistency

### ✅ **14. Style and Standards**
- [ ] Follow established naming conventions (snake_case, etc.)
- [ ] Use consistent indentation and formatting
- [ ] Include appropriate comments for complex logic
- [ ] Ensure imports are organized according to standards
- [ ] Check that section headers use exactly 76 equal signs

### ✅ **15. Error Handling**
- [ ] Test behavior with missing data files
- [ ] Verify graceful handling of insufficient data
- [ ] Check error messages are informative
- [ ] Ensure analysis continues if one glacier fails
- [ ] Test with edge cases (very small datasets, etc.)

## Final Review and Documentation

### ✅ **16. Performance Testing**
- [ ] Run complete analysis with all glaciers
- [ ] Monitor execution time and memory usage
- [ ] Verify all expected outputs are generated
- [ ] Check log output for warnings or errors
- [ ] Confirm statistical results are reasonable

### ✅ **17. Documentation Update**
- [ ] Update file header with final feature list
- [ ] Ensure all methods have accurate docstrings
- [ ] Add inline comments for complex calculations
- [ ] Update configuration comments if needed
- [ ] Document any limitations or assumptions

### ✅ **18. Integration Testing**
- [ ] Test with different quality filter settings
- [ ] Verify compatibility with existing output structure
- [ ] Check that OutputManager integration works correctly
- [ ] Confirm file paths and naming are consistent
- [ ] Test with different visualization configurations

## Deployment Checklist

### ✅ **19. Final Validation**
- [ ] Run analysis one final time from start to finish
- [ ] Review all generated plots for scientific accuracy
- [ ] Check summary statistics match expectations
- [ ] Verify README provides clear explanation of analysis
- [ ] Confirm output organization follows established patterns

### ✅ **20. Version Control and Backup**
- [ ] Save final version with descriptive filename
- [ ] Document any changes from template in comments
- [ ] Create backup of working analysis script
- [ ] Update any analysis documentation or notes
- [ ] Consider committing to version control system

## Common Issues and Solutions

### **Data Loading Problems**
- **Issue**: File not found errors
- **Solution**: Check file paths in CONFIG, verify data file existence
- **Prevention**: Use Path.exists() checks in DataLoader

### **Insufficient Data**
- **Issue**: Empty or very small datasets after filtering
- **Solution**: Relax quality filters, check date ranges, verify data format
- **Prevention**: Log data sizes at each processing step

### **Visualization Errors**
- **Issue**: Plots not displaying correctly
- **Solution**: Check data structure, verify color mapping, review subplot logic
- **Prevention**: Test with simple plots first, add data validation

### **Statistical Calculation Issues**
- **Issue**: NaN or infinite values in statistics
- **Solution**: Add checks for minimum sample sizes, handle edge cases
- **Prevention**: Validate input data ranges and distributions

### **Output Management Problems**
- **Issue**: Files not saved to correct locations
- **Solution**: Verify OutputManager initialization, check file permissions
- **Prevention**: Test output structure early in development

## Best Practices Summary

1. **Start Simple**: Begin with basic functionality and add complexity gradually
2. **Test Frequently**: Run partial tests throughout development process
3. **Use Logging**: Include informative log messages for debugging
4. **Handle Errors**: Always check for and handle potential error conditions
5. **Document Everything**: Write clear docstrings and comments
6. **Follow Patterns**: Use established conventions from existing code
7. **Validate Data**: Check data quality and ranges at each step
8. **Review Output**: Manually inspect generated plots and summaries

## Success Criteria

An analysis script is considered complete when:

- ✅ All three glaciers can be processed without errors
- ✅ High-quality plots are generated and saved correctly
- ✅ Summary statistics are accurate and comprehensive
- ✅ README documentation clearly explains the analysis
- ✅ Code follows established structure and style guidelines
- ✅ Error handling gracefully manages edge cases
- ✅ Logging provides useful progress and debugging information
- ✅ Output organization follows standard directory structure

## Template Files Quick Reference

- **`analysis_template.py`** - Complete code template to copy and modify
- **`CODE_STRUCTURE_REFERENCE.md`** - Detailed implementation guidelines
- **`config_examples.py`** - Configuration patterns and examples
- **`analysis_checklist.md`** - This development checklist (current file)
- **`output_manager.py`** - Utility for standardized output management

Following this checklist ensures that new analysis scripts maintain the same high quality and consistency as existing files in the glacier albedo research codebase.