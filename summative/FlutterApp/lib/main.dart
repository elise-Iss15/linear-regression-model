import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const AkaziScrollApp());
}

class AkaziScrollApp extends StatelessWidget {
  const AkaziScrollApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AKAZI SCROLL',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.indigo, useMaterial3: true),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  static const String apiBaseUrl = 'https://akazi-scroll-api.onrender.com';

  final _viewsController = TextEditingController();

  String? _selectedTitleCategory;
  String? _selectedState;
  String? _selectedWorkType;
  String? _selectedExperienceLevel;

  bool _isLoading = false;
  String? _resultText;
  bool _isError = false;

  final List<String> titleCategories = [
    'Administrative/Office Support',
    'Customer Service/Support',
    'Data/Analytics',
    'Design',
    'Education',
    'Engineering',
    'Finance',
    'General Labor/Hospitality',
    'HR/Recruiting',
    'Healthcare',
    'Legal',
    'Management/Executive',
    'Marketing',
    'Other',
    'Sales',
    'Senior/Lead',
    'Technical/Trades',
  ];

  final List<String> states = [
    'AK',
    'AL',
    'AR',
    'AZ',
    'CA',
    'CO',
    'CT',
    'DC',
    'DE',
    'FL',
    'GA',
    'HI',
    'IA',
    'ID',
    'IL',
    'IN',
    'International/Other',
    'KS',
    'KY',
    'LA',
    'MA',
    'MD',
    'ME',
    'MI',
    'MN',
    'MO',
    'MS',
    'MT',
    'NC',
    'ND',
    'NE',
    'NH',
    'NJ',
    'NM',
    'NV',
    'NY',
    'OH',
    'OK',
    'OR',
    'PA',
    'RI',
    'SC',
    'SD',
    'TN',
    'TX',
    'US - Metro Area (Unspecified State)',
    'US - Unspecified',
    'UT',
    'VA',
    'VT',
    'WA',
    'WI',
    'WV',
    'WY',
  ];

  final List<String> workTypes = [
    'Full-time',
    'Internship',
    'Contract',
    'Part-time',
    'Temporary',
    'Other',
    'Volunteer',
  ];

  final List<String> experienceLevels = [
    'Not Specified',
    'Internship',
    'Entry level',
    'Associate',
    'Mid-Senior level',
    'Director',
    'Executive',
  ];

  Future<void> _predictSalary() async {
    if (_selectedTitleCategory == null ||
        _selectedState == null ||
        _selectedWorkType == null ||
        _selectedExperienceLevel == null ||
        _viewsController.text.isEmpty) {
      setState(() {
        _isError = true;
        _resultText = 'Please fill in all fields before predicting.';
      });
      return;
    }

    final views = int.tryParse(_viewsController.text);
    if (views == null || views < 0 || views > 100000) {
      setState(() {
        _isError = true;
        _resultText = 'Views must be a whole number between 0 and 100,000.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _resultText = null;
      _isError = false;
    });

    try {
      final response = await http.post(
        Uri.parse('$apiBaseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'title_category': _selectedTitleCategory,
          'state': _selectedState,
          'formatted_work_type': _selectedWorkType,
          'formatted_experience_level': _selectedExperienceLevel,
          'views': views,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final salary = data['predicted_salary'];
        setState(() {
          _isError = false;
          _resultText = '\$${salary.toStringAsFixed(2)} ${data['currency']}';
        });
      } else {
        setState(() {
          _isError = true;
          _resultText = 'Error ${response.statusCode}: ${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _isError = true;
        _resultText =
            'Could not reach the server. Note: the free-tier API may take '
            '30-60 seconds to wake up if idle — please try again.';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildDropdown({
    required String label,
    required List<String> items,
    required String? value,
    required void Function(String?) onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: DropdownButtonFormField<String>(
        initialValue: value,
        isExpanded: true,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
        items: items
            .map((item) => DropdownMenuItem(value: item, child: Text(item)))
            .toList(),
        onChanged: onChanged,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AKAZI SCROLL — Salary Estimator')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Estimate a fair salary for a job posting',
                style: TextStyle(fontSize: 16, color: Colors.grey),
              ),
              const SizedBox(height: 16),
              _buildDropdown(
                label: 'Job Category',
                items: titleCategories,
                value: _selectedTitleCategory,
                onChanged: (v) => setState(() => _selectedTitleCategory = v),
              ),
              _buildDropdown(
                label: 'State',
                items: states,
                value: _selectedState,
                onChanged: (v) => setState(() => _selectedState = v),
              ),
              _buildDropdown(
                label: 'Work Type',
                items: workTypes,
                value: _selectedWorkType,
                onChanged: (v) => setState(() => _selectedWorkType = v),
              ),
              _buildDropdown(
                label: 'Experience Level',
                items: experienceLevels,
                value: _selectedExperienceLevel,
                onChanged: (v) => setState(() => _selectedExperienceLevel = v),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: TextField(
                  controller: _viewsController,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    labelText: 'Views (0 - 100,000)',
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _isLoading ? null : _predictSalary,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: _isLoading
                    ? const SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Predict', style: TextStyle(fontSize: 16)),
              ),
              const SizedBox(height: 24),
              if (_resultText != null)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: _isError ? Colors.red.shade50 : Colors.green.shade50,
                    border: Border.all(
                      color: _isError ? Colors.red : Colors.green,
                    ),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _isError ? 'Error' : 'Predicted Salary',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: _isError
                              ? Colors.red.shade900
                              : Colors.green.shade900,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(_resultText!, style: const TextStyle(fontSize: 18)),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
